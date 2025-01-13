# This file contains the TokenIndex class which is used to categorize tokens and get the index of tokens in the string tokens
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal
import random
import yaml


# Load the YAML configuration file
def load_config(yaml_file: str) -> dict:
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

config = load_config("easyroutine/interpretability/config/config.yaml")

SUPPORTED_MODELS = config['models']
SUPPORTED_TOKENS = config['token_position']


class TokenIndex:
    r"""
    TokenIndex is one of the core class of the interpretability module.
    It is used to find the right indexes that correspond to the tokens in the input of the model. 
    In this way we are able to extract the right hidden states and attention weights, based on the tokens we are interested in.
    It support mixed modalities inputs, with both text and images.
    
    """
    def __init__(
        self,
        model_name: str,
        split_positions: Optional[List[int]] = None,
        split_tokens: Optional[List[str]] = None,
    ):
        r"""
        Args:
            model_name: str (required): the name of the model
            split_positions: List[int] (optional): a list of integers that represent the positions where to split the tokens. 
            split_tokens: List[str] (optional): a list of strings that represent the tokens where to split the tokens.
        
        
        The split_positions and split_tokens are mutually exclusive.
        The idea of the split is the following. Immagine to have an input string of tokens like this: ["I", "love", "cats", "and", "dogs". "What", "about", "you?"]
        Then, i want to extract/ablate/intervene on the second sentence. I can do it by specifying the split_positions=[5] or split_tokens=["What"].
        In this way, the tokens will be split in two groups: ["I", "love", "cats", "and"] and ["dogs", "What", "about", "you?"] with names "position-group-0" and "position-group-1".
        """
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
            if (token == start_image_token and not in_image_sequence):
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
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[int]]]:
        if self.split_tokens:
            return self.group_tokens_by_split_tokens(string_tokens)
        elif self.split_positions:
            return self.group_tokens_by_positions(string_tokens)
        else:
            return {0: string_tokens}, {0: list(range(len(string_tokens)))}

    def group_tokens_by_positions(
        self, string_tokens: List[str]
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[int]]]:
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
            positions_group[i] = list(range(positions_group[i][0], positions_group[i][1]))
            

        for i, group in positions_group.items():
            tokens_group[i] = string_tokens[group[0] : group[1]]

        return tokens_group, positions_group

    def group_tokens_by_split_tokens(
        self, string_tokens: List[str]
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[int]]]:
        tokens_group, positions_group = {}, {}
        current_group = 0
        start_pos = 0

        for i, token in enumerate(string_tokens):
            
            if isinstance(self.split_tokens, list) and token in self.split_tokens:
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
        return_type: Literal["list", "dict", "all"] = "list",
    ) -> Union[List[int],  Dict, Tuple[List[int], Dict]]:
        r"""
        Main interface to get the indexes of the tokens in the input string tokens.
        Args:
            tokens: List[str] (required): a list of strings that represent the tokens we are interested in.
            string_tokens: List[str] (required): a list of strings that represent the input tokens.
            return_type: Literal["list", "int", "dict"] (optional): the type of the return value. 
                If "list" it returns a list of integers, if "int" it returns an integer, if "dict" it returns a dictionary.
        
        Returns:
            tokens_positions: Union[List[int], int, Dict]: the indexes of the tokens in the input string tokens in the format specified by return_type.
            
        Supported tokens:
            - `last`: the last token of the input sequence
            - `last-2`: the second last token of the input sequence
            - `last-4`: the fourth last token of the input sequence
            - `last-image`: the last token of the image sequence
            - `end-image`: the end token of the image sequence
            - `all-text`: all the tokens of the text sequence
            - `all`: all the tokens of the input sequence
            - `all-image`: all the tokens of the image sequence
            - `special`: special list of tokens based on the model
            - `random-text`: a random token from the text sequence
            - `random-image`: a random token from the image sequence
            - `random-text-n`: n random tokens from the text sequence
            - `random-image-n`: n random tokens from the image sequence
            - `position-group-i`: the i-th group of tokens based on the split_positions or split_tokens
            - `random-position-group-i`: a random token from the i-th group of tokens based on the split_positions or split_tokens
        
        Examples:
            >>> string_tokens = ["start-image", "img1", "img2", "end-image", I", "love", "cats", "and", "dogs", "What", "about", "you?"]
            >>> tokens = ["end-image", "all-text", "last", "position-group-1", "position-group-2"]
            >>> TokenIndex("facebook/Chameleon-7b", split_tokens = ["cats", "dogs"]).get_token_index(tokens, string_tokens, return_type="dict")
            {'end-image': [3], 'all-text': [4, 5, 6, 7, 8, 9, 10, 11], 'last': [-1], "position-group-1": [7,8], "position-group-2": [9, 10, 11]}
        """
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
        tokens_positions = self.get_tokens_positions(tokens, token_indexes)

        # if return_type == "int":
        #     if len(tokens_positions) > 1:
        #         raise ValueError(
        #             "More than one token requested: return_type should be list, got int"
        #         )
        #     return tokens_positions[0]
        if return_type == "dict":
            return self.get_token_dict(token_indexes)
        if return_type == "all":
            return tokens_positions, self.get_token_dict(token_indexes)
        return tokens_positions

    def get_tokens_positions(
        self, tokens: List[str], token_indexes: Dict[str, List[int]]
    ) -> List[int]:
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
            if token_dict[token] is not None:
                tokens_positions.extend(token_dict[token]) #type: ignore

        return tokens_positions

    def parse_random_group_token(self, token: str) -> Tuple[str, int]:
        group_and_n = token.split("-")[2:]
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
    ) -> Dict[str, Optional[List[int]]]:
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
            "random-text": None if len(token_indexes["text"]) == 0 else [random.choice(token_indexes["text"])],
            "random-image": None if len(token_indexes["image"]) == 0 else [random.choice(token_indexes["image"])],
            "special-pixtral": [1052, 1051, 1038, 991, 1037, 1047],
            **{
                k: v
                for k, v in token_indexes.items()
                if k.startswith("position-group-")
            },
            **random_position_dict,
        }
# from typing import Dict, List, Optional, Union, Literal, Tuple
# import random
# from easyroutine.interpretability.models import SUPPORTED_MODELS, SUPPORTED_TOKENS

# class TokenIndex:
#     def __init__(
#         self,
#         model_name: str,
#         split_positions: Optional[List[int]] = None,
#         split_tokens: Optional[List[str]] = None,
#     ):
#         self.model_name = model_name
#         self.split_tokens = split_tokens
#         self.split_positions = sorted(split_positions) if split_positions else []

#     def find_occurrences(self, lst: List[str], target: str) -> List[int]:
#         return [i for i, x in enumerate(lst) if x == target]

#     def categorize_tokens(self, string_tokens: List[str]) -> Dict[str, List[int]]:
#         if self.model_name not in SUPPORTED_MODELS:
#             raise ValueError("Unsupported model_name")

#         start_image_token, special, end_image_token = SUPPORTED_MODELS[self.model_name]

#         image_start_tokens, image_end_tokens, image_tokens, last_line_image_tokens = (
#             [],
#             [],
#             [],
#             [],
#         )
#         text_tokens, special_tokens = [], []

#         in_image_sequence = False

#         for i, token in enumerate(string_tokens):
#             if token == start_image_token and not in_image_sequence:
#                 in_image_sequence = True
#                 image_start_tokens.append(i)
#             elif in_image_sequence and token == end_image_token:
#                 in_image_sequence = False
#                 image_end_tokens.append(i)
#                 last_line_image_tokens.append(i - 1)
#             elif in_image_sequence and special and token == special:
#                 special_tokens.append(i)
#             elif in_image_sequence:
#                 image_tokens.append(i)
#             elif token == start_image_token:
#             # Handle the case where there is no distinct start and end token
#                 image_tokens.append(i)
#             else:
#                 # Handle the case where there is no distinct start and end token
#                 if len(image_tokens) > 0 and len(image_end_tokens) == 0:
#                     image_tokens.append(i-1)
#                     last_line_image_tokens.append(i-2)
#                 text_tokens.append(i)
            
            

#         tokens_group, positions_group = self.group_tokens(string_tokens)

#         position_dict = {
#             f"position-group-{i}": positions_group[i] for i in positions_group
#         }

#         return {
#             "image_start": image_start_tokens,
#             "image_end": image_end_tokens,
#             "image": image_tokens,
#             "last_line_image": last_line_image_tokens,
#             "text": text_tokens,
#             "special": special_tokens,
#             **position_dict,
#         }

#     def group_tokens(
#         self, string_tokens: List[str]
#     ) -> (Dict[int, List[str]], Dict[int, List[int]]):
#         if self.split_tokens:
#             return self.group_tokens_by_split_tokens(string_tokens)
#         elif self.split_positions:
#             return self.group_tokens_by_positions(string_tokens)
#         else:
#             return {0: string_tokens}, {0: list(range(len(string_tokens)))}

#     def group_tokens_by_positions(
#         self, string_tokens: List[str]
#     ) -> (Dict[int, List[str]], Dict[int, List[int]]):
#         tokens_group, positions_group = {}, {}
#         for i, pos in enumerate(self.split_positions):
#             if i == 0:
#                 positions_group[i] = [0, pos]
#             else:
#                 positions_group[i] = [self.split_positions[i - 1], pos]
#         positions_group[len(self.split_positions)] = [
#             self.split_positions[-1],
#             len(string_tokens),
#         ]

#         # modify the positions_group to include all the indexes and not just the start and end
#         for i in range(len(positions_group)):
#             positions_group[i] = list(
#                 range(positions_group[i][0], positions_group[i][1])
#             )

#         for i, group in positions_group.items():
#             tokens_group[i] = string_tokens[group[0] : group[1]]

#         return tokens_group, positions_group

#     def group_tokens_by_split_tokens(
#         self, string_tokens: List[str]
#     ) -> (Dict[int, List[str]], Dict[int, List[int]]):
#         tokens_group, positions_group = {}, {}
#         current_group = 0
#         start_pos = 0

#         for i, token in enumerate(string_tokens):
#             if token in self.split_tokens:
#                 positions_group[current_group] = [start_pos, i]
#                 tokens_group[current_group] = string_tokens[start_pos:i]
#                 current_group += 1
#                 start_pos = i + 1

#         positions_group[current_group] = [start_pos, len(string_tokens)]
#         tokens_group[current_group] = string_tokens[start_pos:]

#         return tokens_group, positions_group

#     def get_token_index(
#         self,
#         tokens: List[str],
#         string_tokens: List[str],
#         return_type: Literal["list", "int", "dict", "all"] = "all",
#     ) -> Union[List[int], int, Dict]:
#         if not all(
#             token in SUPPORTED_TOKENS
#             or token.startswith("position-group-")
#             or token.startswith("random-position-group-")
#             for token in tokens
#         ):
#             raise ValueError(
#                 f"Unsupported token type: {tokens}. Supported tokens are: {SUPPORTED_TOKENS} and position-group-0, position-group-1, etc or random-position-group-0, random-position-group-1, etc"
#             )

#         # Check if split_positions is required but not provided
#         if self.split_positions is None and any(
#             token.startswith("position-group-")
#             or token.startswith("random-position-group-")
#             for token in tokens
#         ):
#             raise ValueError(
#                 "split_positions cannot be None when a group position token is requested"
#             )

#         token_indexes = self.categorize_tokens(string_tokens)
#         tokens_positions, token_dict = self.get_tokens_positions(tokens, token_indexes)

#         if return_type == "int":
#             if len(tokens_positions) > 1:
#                 raise ValueError(
#                     "More than one token requested: return_type should be list, got int"
#                 )
#             return tokens_positions[0]
#         if return_type == "dict":
#             return token_dict
#         if return_type == "all":
#             return tokens_positions, token_dict
#         return tokens_positions

#     def get_tokens_positions(
#         self, tokens: List[str], token_indexes: Dict[str, List[int]]
#     ) -> Tuple[List[int], Dict]:
#         tokens_positions = []
#         position_dict = {
#             k: v for k, v in token_indexes.items() if k.startswith("position-group-")
#         }
#         random_position_dict = {
#             f"random-{k}": random.sample(v, 1) for k, v in position_dict.items()
#         }

#         for token in tokens:
#             if token.startswith("random-position-group-"):
#                 group, n = self.parse_random_group_token(token)
#                 random_position_dict[token] = random.sample(
#                     position_dict[f"position-group-{group}"], int(n)
#                 )
#             elif token.startswith("random-image"):
#                 n = token.split("-")[-1]
#                 random_position_dict[token] = random.sample(
#                     token_indexes["image"], int(n) if n else 1
#                 )

#         token_dict = self.get_token_dict(token_indexes, random_position_dict)

#         for token in tokens:
#             tokens_positions.extend(token_dict[token])

#         return tokens_positions, token_dict

#     def parse_random_group_token(self, token: str) -> (str, int):
#         group_and_n = token.split("-")[3:]
#         if len(group_and_n) > 1:
#             group, n = group_and_n
#         else:
#             group = group_and_n[0]
#             n = 1
#         return group, int(n)

#     def get_token_dict(
#         self,
#         token_indexes: Dict[str, List[int]],
#         random_position_dict: Dict[str, List[int]] = {},
#     ) -> Dict[str, List[int]]:
#         return {
#             "last": [-1],
#             "last-2": [-2],
#             "last-4": [-4],
#             "last-image": token_indexes["last_line_image"],
#             "end-image": token_indexes["image_end"],
#             "all-text": token_indexes["text"],
#             "all": list(range(len(token_indexes["text"]))),
#             "all-image": token_indexes["image"],
#             "special": token_indexes["special"],
#             "random-text": None
#             if len(token_indexes["text"]) == 0
#             else [random.choice(token_indexes["text"])],
#             "random-image": None
#             if len(token_indexes["image"]) == 0
#             else [random.choice(token_indexes["image"])],
#             "special-pixtral": [1052, 1051, 1038, 991, 1037, 1047],
#             **{
#                 k: v
#                 for k, v in token_indexes.items()
#                 if k.startswith("position-group-")
#             },
#             **random_position_dict,
#         }
