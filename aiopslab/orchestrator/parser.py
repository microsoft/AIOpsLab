# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parser for various APIs that an Agent may invoke."""

import re
import ast

from aiopslab.utils.status import ResponseParsingError


class ResponseParser:
    def __init__(self):
        pass

    def parse(self, response: str) -> dict:
        """Parses the response string to extract the API name and arguments.

        Args:
            response (str): The response string (typically an agent's response).

        Returns:
            dict: The parsed API name and arguments.
        """
        code_block = self.extract_codeblock(response)
        context = self.extract_context(response)
        api_name = self.parse_api_name(code_block)
        args, kwargs = self.parse_args(
            code_block, is_shell_command=api_name == "exec_shell"
        )
        return {
            "api_name": api_name,
            "args": args,
            "kwargs": kwargs,
            "context": context,
        }

    def extract_codeblock(self, response: str) -> str:
        """Extract a markdown code block from a string.

        Args:
            response (str): The response string.

        Returns:
            str: The extracted code block.
        """
        outputlines = response.split("\n")
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if len(indexlines) < 2:
            return ""
        return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])

    def extract_context(self, response: str) -> list:
        """Extract context outside of a code block.

        Args:
            response (str): The response string.

        Returns:
            list: The extracted context.
        """
        pattern = r"(?:```[\s\S]*?```)|(.*?)(?:(?=```)|$)"
        matches = re.findall(pattern, response, re.DOTALL)
        context = [match.strip() for match in matches if match.strip()]

        return context

    def parse_api_name(self, response: str) -> str:
        """Parses the API name from the response function call.

        >>> parse_api_name("get_logs()")
        "get_logs"

        Args:
            response (str): The response string.

        Returns:
            str: The API name.
        """

        first_parenthesis = response.find("(")
        if first_parenthesis != -1:
            return response[:first_parenthesis].strip()
        return ""

    def parse_args(self, response: str, is_shell_command=False) -> list:
        """Parses the arguments of a function call.

        >>> parse_args("get_logs(10, 'error')")
        [10, 'error']

        Args:
            response (str): The response string.

        Returns:
            list: The parsed arguments.
        """
        first_parenthesis = response.find("(")
        last_parenthesis = response.rfind(")")

        if first_parenthesis != -1 and last_parenthesis != -1:
            args_str = response[first_parenthesis + 1 : last_parenthesis].strip()

            # case: no arguments
            if not args_str:
                return [], {}

            # case: single quoted string argument
            if is_shell_command:
                # remove keyword
                if args_str.startswith("command="):
                    args_str = args_str.replace("command=", "").strip()

                if args_str.startswith('"') and args_str.endswith('"'):
                    arg_str = args_str.strip('"')
                elif args_str.startswith("'") and args_str.endswith("'"):
                    arg_str = args_str.strip("'")
                else:
                    raise ResponseParsingError(
                        "Error when parsing response: commands must be quoted strings"
                    )

                arg_str = arg_str.replace('\\"', '"').replace("\\'", "'")
                return [arg_str], {}

            # case: positional/kwargs handled w/ ast.parse
            try:
                parsed = ast.parse(f"func({args_str})")
                call = parsed.body[0].value
                args, kwargs = [], {}

                for arg in call.args:
                    if isinstance(arg, ast.Constant):
                        args.append(arg.value)
                    elif isinstance(arg, (ast.List, ast.Tuple)):
                        args.append([self.eval_ast_node(elt) for elt in arg.elts])
                    elif isinstance(arg, ast.Dict):
                        args.append(
                            {
                                self.eval_ast_node(key): self.eval_ast_node(value)
                                for key, value in zip(arg.keys, arg.values)
                            }
                        )
                    else:
                        args.append(self.eval_ast_node(arg))

                for kwarg in call.keywords:
                    kwargs[kwarg.arg] = self.eval_ast_node(kwarg.value)

                return args, kwargs
            except Exception as e:
                raise ResponseParsingError(f"Error parsing response: {str(e)}")

        raise ResponseParsingError("No API call found!")

    def eval_ast_node(self, node):
        """Evaluates an AST node to its Python value."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.List):
            return [self.eval_ast_node(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self.eval_ast_node(key): self.eval_ast_node(value)
                for key, value in zip(node.keys, node.values)
            }
        elif isinstance(node, ast.Name):
            if node.id == "True":
                return True
            elif node.id == "False":
                return False
            elif node.id == "None":
                return None
        raise ValueError(f"Unsupported AST node type: {type(node)}")
