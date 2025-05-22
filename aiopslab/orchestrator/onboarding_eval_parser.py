"""Custom parser for the onboarding task evaluator"""

import re
import ast

from aiopslab.utils.status import ResponseParsingError

class EvalParser:
    def __init__(self):
        # Define list of known API commands that need special handling
        self.known_apis = ["submit"]
        
    def parse(self, response: str) -> dict:
        """Parses the response string to extract the API name and arguments.

        Args:
            response (str): The response string (typically an agent's response).

        Returns:
            dict: The parsed API name and arguments.
        """
        code_block = self.extract_codeblock(response)
        context = self.extract_context(response)
        
        # If there's no code block, check if the response itself is a command
        if not code_block:
            code_block = response.strip()
            
        # Check if the code block is a simple "submit" command without parameters
        if code_block.strip() == "submit":
            return {
                "api_name": "submit",
                "args": [None],  # Placeholder argument 
                "kwargs": {},
                "context": context,
            }
            
        # Handle other known APIs with function call syntax
        if any(code_block.strip().startswith(api + "(") for api in self.known_apis):
            api_name = self.parse_api_name(code_block)
            args, kwargs = self.parse_args(code_block)
            return {
                "api_name": api_name,
                "args": args,
                "kwargs": kwargs,
                "context": context,
            }
            
        # Default to exec_shell for unrecognized commands
        # Strip any leading/trailing backticks if present
        command = code_block.strip("` \n")
        return {
            "api_name": "exec_shell",
            "args": [command],
            "kwargs": {},
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

        Args:
            response (str): The response string.

        Returns:
            str: The API name.
        """
        first_parenthesis = response.find("(")
        if first_parenthesis != -1:
            return response[:first_parenthesis].strip()
        return ""

    def parse_args(self, response: str) -> tuple:
        """Parses the arguments of a function call.

        Args:
            response (str): The response string.

        Returns:
            tuple: (args, kwargs) - Lists of positional and keyword arguments.
        """
        first_parenthesis = response.find("(")
        last_parenthesis = response.rfind(")")

        if first_parenthesis != -1 and last_parenthesis != -1:
            args_str = response[first_parenthesis + 1 : last_parenthesis].strip()

            # case: no arguments
            if not args_str:
                return [], {}

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
                raise ResponseParsingError(f"Error parsing arguments: {str(e)}")

        return [], {}

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