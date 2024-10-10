import dspy
import textwrap
from typing import List
import ast
import re
import logging
from zero_sum_eval.player import Player
from zero_sum_eval.registry import PLAYER_REGISTRY, METRIC_REGISTRY

logger = logging.getLogger('ZeroSumEval')



@METRIC_REGISTRY.register("pyjail_code_validation_metric")
def validate_pyjail_code(example, prediction, trace=None):
    def _verify_code_has_function(self, tree, name):

        class functionVisitor(ast.NodeVisitor):

            def visit_FunctionDef(self, node):
                if node.name == name:
                    if len(node.args.args) == 1:
                        self.found_jail = True

                self.generic_visit(node)

    
    match = re.search(r'###START(.*?)###END', move.decode('utf-8'), re.DOTALL)
    code = match.group(1).strip()
    if not match:
        return 0
    try:

        move = textwrap.dedent(move)

        tree = ast.parse(move)
        
        code_has_func = self._verify_code_has_function(tree, 'jail')
        
        if code_has_func != None:
            raise ValueError(code_has_func)
    except Exception:
        return 0


@METRIC_REGISTRY.register("pyjail_solution_validation_metric")
def validate_pyjail_solve(example, prediction, trace=None):
    match = re.search(r'###START(.*?)###END', move.decode('utf-8'), re.DOTALL)
    code = match.group(1).strip()
    try:

        move = textwrap.dedent(move)

        ast.parse(move)
        
        return 1

    except Exception:
        return 0

class GenerateCode(dspy.Signature):
    """Generate PyJail code that makes access harder to the FLAG environment variable"""
    role = dspy.InputField(desc="roles of the player")
    message = dspy.InputField(desc="message from the game manager")
    history = dspy.InputField()
    code = dspy.OutputField(desc="PyJail code to limit access to FLAG, start with ###START, end with ###END")

class SolveCode(dspy.Signature):
    """Given PyJail code, generate a solution to access the FLAG environment variable"""
    role = dspy.InputField(desc="roles of the player") 
    message = dspy.InputField(desc="message from the game manager")
    pyjail_code = dspy.InputField(desc="PyJail code to solve")
    history = dspy.InputField()
    code = dspy.OutputField(desc="Solution code to access FLAG start with ###START, end with ###END")

class GeneratePyjailCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot_generate = dspy.ChainOfThought(GenerateCode)

    def forward(self, role, message, history):
        cot_out = self.cot_generate(role=role, message=message, history=f"{history}")
        return cot_out
    
        

class SolvePyjailCoT(dspy.Module):
    def __init__(self, exlcude = None):
        super().__init__()
        self.cot_solve = dspy.ChainOfThought(SolveCode)

    def forward(self, role, message,  history, pyjail_code = 'source hidden for challenge'):        
        cot_out = self.cot_solve(role=role, message=message, pyjail_code=pyjail_code, history=f"{history}")
        return cot_out
            

class PyjailGeneratorModule(dspy.Module):
    def __init__(self, roles, **kwargs):
        super().__init__()
        self.module_dict = dict()
        for role in roles:
            if role == "DefenderGenerateCode":
                 self.module_dict[role] = GeneratePyjailCoT()
            elif role == "DefenderSolveCode":
                self.module_dict[role] = SolvePyjailCoT()
    
    def forward(self, **kwargs):
        role = kwargs.get('role', None) 
        return self.module_dict[role](**kwargs)

class PyjailPlayerModule(dspy.Module):
    def __init__(self, roles, **kwargs):
        super().__init__()
        self.module_dict = {roles[0]: SolvePyjailCoT()}
    
    def forward(self, **kwargs):
        role = kwargs.get('role', None)
        return self.module_dict[role](**kwargs)


@PLAYER_REGISTRY.register("pyjail", "pyjail_generator")
class DefenderGenerateCode(Player):
    def _build_module(self, roles, **kwargs):        
        return PyjailGeneratorModule(roles, **kwargs)

    def _make_move(self, **kwargs):
        trace = self.module(**kwargs)
        return str(trace.code), trace
        

@PLAYER_REGISTRY.register("pyjail", "pyjail_player")
class PyjailPlayer(Player):
    def _build_module(self, roles, **kwargs):
        return PyjailPlayerModule(roles, **kwargs)

    def _make_move(self, **kwargs):
        trace = self.module(**kwargs) 
        return str(trace.code), trace


