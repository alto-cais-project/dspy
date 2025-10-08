# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 02:33:52 2023

@author: Shahir
"""

import os
import abc
import datetime
import time
from typing import Any, Self

import ujson as json  # type: ignore

import alto


DEFAULT_ENCODING: str = 'utf-8'

system_time = time.monotonic

class BytesSerializable():
    @abc.abstractmethod
    def to_bytes(
        self: Self
    ) -> bytes:

        pass

    @classmethod
    @abc.abstractmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        pass


class JSONDictSerializable(BytesSerializable):
    def __eq__(
        self: Self,
        other: object
    ) -> bool:

        if not isinstance(other, JSONDictSerializable):
            return False

        return self.to_dict() == other.to_dict()  # type: ignore

    def __str__(
        self: Self
    ) -> str:

        return str(self.to_dict())  # type: ignore

    def __repr__(
        self: Self
    ) -> str:

        return str(self.to_dict())  # type: ignore

    @abc.abstractmethod
    def to_dict(
        self: Self
    ) -> dict[str, Any]:

        pass

    def to_json(
        self: Self
    ) -> str:

        return json.dumps(self.to_dict())  # type: ignore

    def to_bytes(
        self: Self
    ) -> bytes:

        return self.to_json().encode(encoding=DEFAULT_ENCODING)

    def to_file(
        self: Self,
        fname: str
    ) -> None:

        with open(fname, 'w') as f:
            json.dump(self.to_dict(), f)  # type: ignore

    @classmethod
    @abc.abstractmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, Any]
    ) -> Self:

        pass

    @classmethod
    def from_json(
        cls: type[Self],
        data: str
    ) -> Self:

        return cls.from_dict(json.loads(data))

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        return cls.from_json(buf.decode(encoding=DEFAULT_ENCODING))

    @classmethod
    def from_file(
        cls: type[Self],
        fname: str
    ) -> Self:

        with open(fname, 'r') as f:
            data = cls.from_json(f.read())

        return data


def number_menu(
    option_list: list[str]
) -> tuple[int, str]:

    print("-" * 60)
    for n in range(len(option_list)):
        print(n, ": ", option_list[n])

    choice = input("Choose the number corresponding to your choice: ")
    for n in range(5):
        try:
            index = int(choice)
            if index < 0 or index > len(option_list) - 1:
                raise ValueError
            print("-" * 60 + "\n")

            return index, option_list[index]

        except ValueError:
            choice = input("Invalid input, choose again: ")

    raise ValueError("Not recieving a valid input")


def get_rel_pkg_path(
    path: str
) -> str:

    return os.path.abspath(os.path.join(os.path.dirname(alto_lib.__file__), "..", path))  # type: ignore


def load_rel_config_json(
    fname: str
) -> dict[str, Any]:

    fname = get_rel_pkg_path(fname)
    with open(fname, 'r') as f:
        data = json.load(f)

    return data


def get_timestamp_str(
    include_seconds: bool = True
) -> str:

    if include_seconds:
        return datetime.datetime.now().strftime("%m-%d-%Y %I-%M-%S %p")
    else:
        return datetime.datetime.now().strftime("%m-%d-%Y %I-%M %p")
