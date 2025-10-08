import ast
import sys
import re
from typing import Dict, Set, Optional, List

# --- Data structures ---
class ModuleInfo:
    """Stores metadata about a module/class in the pipeline."""
    def __init__(self, name: str):
        self.name = name
        self.input_granularity = None  # Expected input split (Word/Line/Paragraph)
        self.output_split = None       # Output type of the module
        self.children: Set[str] = set()  # Downstream modules
        self.propagates_output: Optional[bool] = None  # True if output depends on input


# --- Split Type Mapping ---
SPLIT_TYPE_MAP = {
    "TokenSplit": "WordOutput",
    "WordSplit": "WordOutput",
    "NewlineSplit": "LineOutput",
    "LineSplit": "LineOutput",
    "ParagraphSplit": "ParagraphOutput",
}

def map_split_to_output_type(split_str: str) -> str:
    """Normalize split type string to output type."""
    clean_split = split_str.strip("'\"")
    result = (
        SPLIT_TYPE_MAP.get(split_str)
        or SPLIT_TYPE_MAP.get(clean_split)
        or SPLIT_TYPE_MAP.get(f"'{clean_split}'")
    )
    return result or "LineOutput"  # Default fallback


# --- AST Analyzer ---
class DataFlowAnalyzer(ast.NodeVisitor):
    """Analyzes data flow between modules in the AST."""
    def __init__(self):
        self.class_infos: Dict[str, ModuleInfo] = {}
        self.current_class = None
        self.var_producers: Dict[str, str] = {}  # Tracks variable -> producer mapping
        self.edges = []  # (parent, child) relationships
        self.instance_map: Dict[str, str] = {}   # Tracks instance variable -> class
        self.tree = None
        self.defined_classes: Set[str] = set()  # All class names in file

    def visit_ClassDef(self, node: ast.ClassDef):
        """Record class and its decorators (input granularity)."""
        self.current_class = node.name
        self.defined_classes.add(node.name)
        self.class_infos.setdefault(node.name, ModuleInfo(node.name))

        # Parse @input_granularity decorators
        for deco in node.decorator_list:
            if isinstance(deco, ast.Call) and getattr(deco.func, "id", None) == "input_granularity":
                granularity_info = {}
                for kw in deco.keywords:
                    if kw.arg in ["split", "level"]:
                        granularity_info["split"] = ast.unparse(kw.value)
                    elif kw.arg == "arg":
                        granularity_info["arg"] = ast.unparse(kw.value)
                self.class_infos[node.name].input_granularity = granularity_info

        self.generic_visit(node)
        self.current_class = None

    def visit_Assign(self, node: ast.Assign):
        """Detect assignments like X = StreamingTextProducer(output=...)."""
        if isinstance(node.value, ast.Call) and getattr(node.value.func, "id", None) == "StreamingTextProducer":
            if isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                self.class_infos.setdefault(var_name, ModuleInfo(var_name))

                for kw in node.value.keywords:
                    if kw.arg == "output":
                        val = kw.value
                        self.class_infos[var_name].output_split = val.id
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await):
        """Track await calls for asynchronous producers."""
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == "acall":
                producer = ast.unparse(func.value)
                target = self._get_assign_target(node)
                if target:
                    self.var_producers[target] = producer

    def visit_Call(self, node: ast.Call):
        """Detect method calls and pmap calls to create data flow edges."""
        func_name = ast.unparse(node.func)
        
        # Detect method calls like self.sqg_module.aforward()
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ["aforward", "forward"]:
                # This is a method call, track it as a producer
                target = self._get_assign_target(node)
                if target:
                    # Get the module name from the attribute access
                    module_name = ast.unparse(node.func.value)
                    # Resolve through instance mapping
                    resolved_module = self.instance_map.get(module_name, module_name)
                    self.var_producers[target] = resolved_module
                    print(f"Detected method call: {module_name}.{node.func.attr} -> {target} (resolved: {resolved_module})")
        
        # Detect pmap calls and create edges (producer -> consumer)
        if func_name.endswith("pmap") and len(node.args) >= 2:
            iterable = ast.unparse(node.args[0])
            consumer = ast.unparse(node.args[1])

            if iterable in self.var_producers:
                producer = self.var_producers[iterable]
                self._add_edge(producer, consumer)
                print(f"Detected pmap: {producer} -> {consumer}")

        self.generic_visit(node)

    def _add_edge(self, parent: str, child: str):
        """Add a parent -> child relationship in class infos and edges."""
        self.class_infos.setdefault(parent, ModuleInfo(parent))
        self.class_infos.setdefault(child, ModuleInfo(child))
        self.class_infos[parent].children.add(child)
        self.edges.append((parent, child))

    def _get_assign_target(self, node):
        """Find the variable being assigned in an expression."""
        parent = getattr(node, "parent", None)
        if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
            if isinstance(parent.targets[0], ast.Name):
                return parent.targets[0].id
        return None

    def resolve_constructor_calls(self):
        """Map instance variables to actual classes based on constructor calls."""
        print(f"Defined classes found: {self.defined_classes}")
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                class_name = node.func.id
                if class_name in self.defined_classes:
                    print(f"Found constructor call for class: {class_name}")

                    # Handle keyword arguments
                    for kw in node.keywords:
                        if kw.arg and isinstance(kw.value, ast.Call):
                            if isinstance(kw.value.func, ast.Name):
                                actual_class = kw.value.func.id
                                instance_var = f"self.{kw.arg}"
                                self.instance_map[instance_var] = actual_class
                                print(f"  Mapping: {instance_var} -> {actual_class}")

                    # Handle positional args
                    for i, arg in enumerate(node.args):
                        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                            actual_class = arg.func.id
                            instance_var = f"self.arg_{i}"
                            self.instance_map[instance_var] = actual_class
                            print(f"  Mapping positional arg {i}: {instance_var} -> {actual_class}")

        # Update edges using instance mapping
        resolved_edges = []
        for parent, child in self.edges:
            resolved_parent = self.instance_map.get(parent, parent)
            resolved_child = self.instance_map.get(child, child)
            resolved_edges.append((resolved_parent, resolved_child))

            self.class_infos.setdefault(resolved_parent, ModuleInfo(resolved_parent))
            self.class_infos.setdefault(resolved_child, ModuleInfo(resolved_child))
            self.class_infos[resolved_parent].children.add(resolved_child)

        self.edges = resolved_edges
        print(f"Instance mapping: {self.instance_map}")

    def expand_composite_modules(self):
        """Expand composite modules to find their constituent modules with @input_granularity."""
        print("=== Expanding Composite Modules ===")
        expanded_edges = []
        
        for parent, child in self.edges:
            # Check if child is a composite module that contains modules with @input_granularity
            if child in self.class_infos:
                child_info = self.class_infos[child]
                # Find modules that this composite module contains
                constituent_modules = self._find_constituent_modules(child)
                print(f"Composite module {child} contains: {constituent_modules}")
                
                if constituent_modules:
                    # Create edges from parent to each constituent module
                    for constituent in constituent_modules:
                        expanded_edges.append((parent, constituent))
                        self.class_infos[parent].children.add(constituent)
                        print(f"  Expanded edge: {parent} -> {constituent}")
                else:
                    # Keep original edge if no constituents found
                    expanded_edges.append((parent, child))
            else:
                expanded_edges.append((parent, child))
        
        self.edges = expanded_edges

    def _find_constituent_modules(self, composite_module: str) -> List[str]:
        """Find modules with @input_granularity that are contained in a composite module."""
        constituents = []
        
        # Look through the instance mapping to find modules that belong to this composite
        for instance_var, actual_class in self.instance_map.items():
            # Check if this instance variable belongs to the composite module
            if instance_var.startswith(f"self.") and actual_class in self.class_infos:
                # Check if the actual class has @input_granularity decorator
                class_info = self.class_infos[actual_class]
                if class_info.input_granularity:
                    constituents.append(actual_class)
                    print(f"  Found constituent with @input_granularity: {actual_class}")
        
        return constituents


# --- Constructor Analyzer ---
class ConstructorAnalyzer(ast.NodeVisitor):
    """Dedicated analyzer to map constructor calls to instance variables."""
    
    def __init__(self, defined_classes: Set[str]):
        self.defined_classes = defined_classes
        self.constructor_calls = []
        self.instance_mappings = {}
    
    def visit_Call(self, node: ast.Call):
        """Record constructor calls of known classes."""
        if isinstance(node.func, ast.Name) and node.func.id in self.defined_classes:
            call_info = {
                'class_name': node.func.id,
                'keyword_mappings': {},
                'positional_mappings': {}
            }
            
            # Keyword args
            for kw in node.keywords:
                if kw.arg and isinstance(kw.value, ast.Call) and isinstance(kw.value.func, ast.Name):
                    call_info['keyword_mappings'][kw.arg] = kw.value.func.id
            
            # Positional args
            for i, arg in enumerate(node.args):
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                    call_info['positional_mappings'][i] = arg.func.id
            
            self.constructor_calls.append(call_info)
        
        self.generic_visit(node)
    
    def get_instance_mappings(self) -> Dict[str, str]:
        """Convert constructor calls to instance variable -> class mapping."""
        mappings = {}
        for call in self.constructor_calls:
            # Keyword args
            for param_name, actual_class in call['keyword_mappings'].items():
                mappings[f"self.{param_name}"] = actual_class
            # Positional args
            for pos, actual_class in call['positional_mappings'].items():
                mappings[f"self.arg_{pos}"] = actual_class
        return mappings


# --- Dependency Analysis ---
class DependencyAnalyzer(ast.NodeVisitor):
    """Detects if a function's return depends on its input arguments."""
    def __init__(self, input_args: Set[str]):
        self.dependent_vars = set(input_args)
        self.returns_dependent = False

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if self._expr_uses_dependent(node.value):
                self.dependent_vars.add(target)
        self.generic_visit(node)

    def visit_Return(self, node):
        if node.value and self._expr_uses_dependent(node.value):
            self.returns_dependent = True

    def _expr_uses_dependent(self, expr):
        """Check recursively if expression uses dependent vars."""
        if isinstance(expr, ast.Name):
            return expr.id in self.dependent_vars
        for child in ast.iter_child_nodes(expr):
            if self._expr_uses_dependent(child):
                return True
        return False


# --- Consumer Object Generation ---
def generate_consumer_objects(class_infos: Dict[str, ModuleInfo], start_module: str) -> List[str]:
    """Generate consumer objects for downstream modules."""
    if start_module not in class_infos:
        return []

    start_info = class_infos[start_module]
    producer_output_split = start_info.output_split or "LineOutput"

    consumer_objects = []
    uid = 1
    for child_name in start_info.children:
        child_info = class_infos.get(child_name)
        if not child_info:
            continue

        # Only generate consumers for modules with @input_granularity decorators
        if not child_info.input_granularity:
            print(f"Skipping {child_name} - no @input_granularity decorator")
            continue

        # Determine processor type for consumer
        processor_type = producer_output_split
        if child_info.input_granularity:
            required_split = child_info.input_granularity.get("split")
            if required_split and required_split != "None":
                req = str(required_split).strip("'\"")
                processor_type = map_split_to_output_type(req)

        consumer_obj = f'ConsumerObject("{child_name}", {processor_type}, {producer_output_split}, uid={uid})'
        consumer_objects.append(consumer_obj)
        uid += 1

    return consumer_objects


# --- Main Analysis Function ---
def analyze_and_generate_consumers(filepath: str, start_module: str):
    """Parse a file, analyze dependencies, and generate consumer objects."""
    with open(filepath, "r") as f:
        source = f.read()
    tree = ast.parse(source)

    # Add parent references to all AST nodes
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    # Initialize analyzer and collect classes
    analyzer = DataFlowAnalyzer()
    analyzer.tree = tree
    analyzer.visit(tree)

    # Resolve constructor calls
    constructor_analyzer = ConstructorAnalyzer(analyzer.defined_classes)
    constructor_analyzer.visit(tree)
    analyzer.instance_map = constructor_analyzer.get_instance_mappings()
    analyzer.resolve_constructor_calls()

    # Expand composite modules to find constituent modules
    analyzer.expand_composite_modules()

    # Dependency analysis for each class's forward methods
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            if class_name in analyzer.class_infos:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name in {"aforward", "forward"}:
                        input_args = {arg.arg for arg in item.args.args if arg.arg != "self"}
                        dep_analyzer = DependencyAnalyzer(input_args)
                        dep_analyzer.visit(item)
                        analyzer.class_infos[class_name].propagates_output = dep_analyzer.returns_dependent

    # Print dependency graph
    print("=== Dependency Graph ===")
    for parent, child in analyzer.edges:
        print(f"{parent} -> {child}")
    print()

    # Generate consumer objects
    consumer_objects = generate_consumer_objects(analyzer.class_infos, start_module)
    print("=== Generated Consumer Objects ===")
    print("consumers = [")
    for consumer_obj in consumer_objects:
        print(f"    {consumer_obj},")
    print("]")

    return consumer_objects


# --- Main runner ---
def main(filepath: str, start_module: str):
    return analyze_and_generate_consumers(filepath, start_module)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python new_ast.py <dspy_app.py> <StartModuleName>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
