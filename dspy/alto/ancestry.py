from typing import List
from alto import ancestry_pb2
import copy

class AncestryId(object):
    def __init__(self, val: int, scope: int):
        self._id = val
        self._scope = scope 

    @classmethod
    def from_proto(cls, proto: ancestry_pb2.AncestryId) -> 'AncestryId':
        return cls(proto.val, proto.scope)

    def to_proto(self) -> ancestry_pb2.AncestryId:
        proto = ancestry_pb2.AncestryId()
        proto.val = self._id
        proto.scope = self._scope
        return proto

    def __str__(self):
        return f"[id: {self._id}, scope: {self._scope}]"
     
    @property
    def value(self):
        return self._id

    @property
    def scope(self):
        return self._scope

    @scope.setter
    def scope(self, val: int):
        self._scope = val

    def __eq__(self, other) -> bool:
        if not isinstance(other, AncestryId):
            return False
        return self._id == other._id and self._scope == other._scope
    
    def __hash__(self):
        return hash((self._id, self._scope))

class AncestryTag(object):
    def __init__(self, values: List[AncestryId] = []):
        self._values = values
        self._is_eos = False
        self._sticky_routing_hash = -1

    @property
    def is_eos(self):
        return self._is_eos

    @is_eos.setter
    def is_eos(self, val: bool):
        self._is_eos = val

    @property
    def sticky_routing_hash(self):
        return self._sticky_routing_hash

    @sticky_routing_hash.setter
    def sticky_routing_hash(self, val: int):
        self._sticky_routing_hash = val

    @classmethod
    def from_proto(cls, proto: ancestry_pb2.AltoMetadata) -> 'AncestryTag':
        ins = cls([AncestryId.from_proto(x) for x in proto.ancestry_tag])
        ins.is_eos = proto.is_eos
        ins.sticky_routing_hash = proto.sticky_routing_hash
        return ins

    @classmethod
    def from_bytes(cls, data: bytes) -> 'AncestryTag':
        try:
            proto = ancestry_pb2.AltoMetadata()
            proto.ParseFromString(data)
            return cls.from_proto(proto)
        except:
            raise ValueError("Invalid data to deserialize into AltoMetadata")

    def to_bytes(self) -> bytes:
        return self.to_proto().SerializeToString()

    def to_proto(self) -> ancestry_pb2.AltoMetadata:
        """Converts the object to a proto object.

        Note that by default it sets the message type to be DATA.
        Caller needs to manually change it if they want to set it as another
        type of message.
        """
        proto = ancestry_pb2.AltoMetadata()
        proto.ancestry_tag.extend([x.to_proto() for x in self._values])
        proto.is_eos = self._is_eos
        proto.sticky_routing_hash = self._sticky_routing_hash
        return proto

    @property
    def values(self):
        return self._values

    def csv_str(self):
        tag = "(["
        for i, val in enumerate(self._values):
            tag += f"{(val.value,val.scope)}"
            if i < len(self._values) - 1:
                tag += ","
        tag += f",{self.is_eos}])"
        return tag


    def __str__(self):
        tag = "["
        for i, val in enumerate(self._values):
            tag += f"{i}: "
            tag += str(val)
            if i < (len(self._values) - 1):
                tag += ","
        tag += "]"
        return tag
    def __repr__(self):
        return str(self)

    def push(self, val: AncestryId) -> 'AncestryTag':
        copied = copy.deepcopy(self._values)
        copied.append(val)
        ans = AncestryTag(copied)
        ans.sticky_routing_hash = self.sticky_routing_hash
        return ans

    def pop(self) -> tuple['AncestryTag', 'AncestryId']:
        copied = copy.deepcopy(self._values)
        last_id = copied.pop()
        ans = AncestryTag(copied)
        ans.sticky_routing_hash = self.sticky_routing_hash
        return (ans, last_id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, AncestryTag):
            return False
        return self._values == other._values and \
                self._sticky_routing_hash == other._sticky_routing_hash
    def __hash__(self):
        return hash(tuple(self._values + [self._sticky_routing_hash]))

    def is_empty(self) -> bool:
        return len(self._values) == 0
