from typing import Dict, List, Optional, Union, Literal, Tuple
import random
from easyroutine.interpretability.models import SUPPORTED_MODELS, SUPPORTED_TOKENS

class TokenIndex:
    def __init__(
        self,
        model_name: str,
        split_positions: Optional[List[int]] = None,
        split_tokens: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.split_tokens = split_tokens
        self.split_positions = sorted(split_positions) if split_positions else []

    def find_occurrences(self, lst: List[str], target: str) -> List[int]:
        return [i for i, x in enumerate(lst) if x == target]

    def categorize_tokens(self, string_tokens: List[str]) -> Dict[str, List[int]]:
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError("Unsupported model_name")

        start_image_token, special, end_image_token = SUPPORTED_MODELS[self.model_name]

        image_start_tokens, image_end_tokens, image_tokens, last_line_image_tokens = (
            [],
            [],
            [],
            [],
        )
        text_tokens, special_tokens = [], []

        in_image_sequence = False

        for i, token in enumerate(string_tokens):
            if token == start_image_token and not in_image_sequence:
                in_image_sequence = True
                image_start_tokens.append(i)
            elif in_image_sequence and token == end_image_token:
                in_image_sequence = False
                image_end_tokens.append(i)
                last_line_image_tokens.append(i - 1)
            elif in_image_sequence and special and token == special:
                special_tokens.append(i)
            elif in_image_sequence:
                image_tokens.append(i)
            else:
                text_tokens.append(i)

        tokens_group, positions_group = self.group_tokens(string_tokens)

        position_dict = {
            f"position-group-{i}": positions_group[i] for i in positions_group
        }

        return {
            "image_start": image_start_tokens,
            "image_end": image_end_tokens,
            "image": image_tokens,
            "last_line_image": last_line_image_tokens,
            "text": text_tokens,
            "special": special_tokens,
            **position_dict,
        }

    def group_tokens(
        self, string_tokens: List[str]
    ) -> (Dict[int, List[str]], Dict[int, List[int]]):
        if self.split_tokens:
            return self.group_tokens_by_split_tokens(string_tokens)
        elif self.split_positions:
            return self.group_tokens_by_positions(string_tokens)
        else:
            return {0: string_tokens}, {0: list(range(len(string_tokens)))}

    def group_tokens_by_positions(
        self, string_tokens: List[str]
    ) -> (Dict[int, List[str]], Dict[int, List[int]]):
        tokens_group, positions_group = {}, {}
        for i, pos in enumerate(self.split_positions):
            if i == 0:
                positions_group[i] = [0, pos]
            else:
                positions_group[i] = [self.split_positions[i - 1], pos]
        positions_group[len(self.split_positions)] = [
            self.split_positions[-1],
            len(string_tokens),
        ]

        # modify the positions_group to include all the indexes and not just the start and end
        for i in range(len(positions_group)):
            positions_group[i] = list(
                range(positions_group[i][0], positions_group[i][1])
            )

        for i, group in positions_group.items():
            tokens_group[i] = string_tokens[group[0] : group[1]]

        return tokens_group, positions_group

    def group_tokens_by_split_tokens(
        self, string_tokens: List[str]
    ) -> (Dict[int, List[str]], Dict[int, List[int]]):
        tokens_group, positions_group = {}, {}
        current_group = 0
        start_pos = 0

        for i, token in enumerate(string_tokens):
            if token in self.split_tokens:
                positions_group[current_group] = [start_pos, i]
                tokens_group[current_group] = string_tokens[start_pos:i]
                current_group += 1
                start_pos = i + 1

        positions_group[current_group] = [start_pos, len(string_tokens)]
        tokens_group[current_group] = string_tokens[start_pos:]

        return tokens_group, positions_group

    def get_token_index(
        self,
        tokens: List[str],
        string_tokens: List[str],
        return_type: Literal["list", "int", "dict", "all"] = "all",
    ) -> Union[List[int], int, Dict]:
        if not all(
            token in SUPPORTED_TOKENS
            or token.startswith("position-group-")
            or token.startswith("random-position-group-")
            for token in tokens
        ):
            raise ValueError(
                f"Unsupported token type: {tokens}. Supported tokens are: {SUPPORTED_TOKENS} and position-group-0, position-group-1, etc or random-position-group-0, random-position-group-1, etc"
            )

        # Check if split_positions is required but not provided
        if self.split_positions is None and any(
            token.startswith("position-group-")
            or token.startswith("random-position-group-")
            for token in tokens
        ):
            raise ValueError(
                "split_positions cannot be None when a group position token is requested"
            )

        token_indexes = self.categorize_tokens(string_tokens)
        tokens_positions, token_dict = self.get_tokens_positions(tokens, token_indexes)

        if return_type == "int":
            if len(tokens_positions) > 1:
                raise ValueError(
                    "More than one token requested: return_type should be list, got int"
                )
            return tokens_positions[0]
        if return_type == "dict":
            return token_dict
        if return_type == "all":
            return tokens_positions, token_dict
        return tokens_positions

    def get_tokens_positions(
        self, tokens: List[str], token_indexes: Dict[str, List[int]]
    ) -> Tuple[List[int], Dict]:
        tokens_positions = []
        position_dict = {
            k: v for k, v in token_indexes.items() if k.startswith("position-group-")
        }
        random_position_dict = {
            f"random-{k}": random.sample(v, 1) for k, v in position_dict.items()
        }

        for token in tokens:
            if token.startswith("random-position-group-"):
                group, n = self.parse_random_group_token(token)
                random_position_dict[token] = random.sample(
                    position_dict[f"position-group-{group}"], int(n)
                )
            elif token.startswith("random-image"):
                n = token.split("-")[-1]
                random_position_dict[token] = random.sample(
                    token_indexes["image"], int(n) if n else 1
                )

        token_dict = self.get_token_dict(token_indexes, random_position_dict)

        for token in tokens:
            tokens_positions.extend(token_dict[token])

        return tokens_positions, token_dict

    def parse_random_group_token(self, token: str) -> (str, int):
        group_and_n = token.split("-")[3:]
        if len(group_and_n) > 1:
            group, n = group_and_n
        else:
            group = group_and_n[0]
            n = 1
        return group, int(n)

    def get_token_dict(
        self,
        token_indexes: Dict[str, List[int]],
        random_position_dict: Dict[str, List[int]] = {},
    ) -> Dict[str, List[int]]:
        return {
            "last": [-1],
            "last-2": [-2],
            "last-4": [-4],
            "last-image": token_indexes["last_line_image"],
            "end-image": token_indexes["image_end"],
            "all-text": token_indexes["text"],
            "all": list(range(len(token_indexes["text"]))),
            "all-image": token_indexes["image"],
            "special": token_indexes["special"],
            "random-text": None
            if len(token_indexes["text"]) == 0
            else [random.choice(token_indexes["text"])],
            "random-image": None
            if len(token_indexes["image"]) == 0
            else [random.choice(token_indexes["image"])],
            "special-pixtral": [1052, 1051, 1038, 991, 1037, 1047],
            **{
                k: v
                for k, v in token_indexes.items()
                if k.startswith("position-group-")
            },
            **random_position_dict,
        }
