import dspy
from zero_sum_eval.core.player import Player
from zero_sum_eval.core.registry import PLAYER_REGISTRY, METRIC_REGISTRY

# Player keys
TEACHER_KEY = "teacher"
STUDENT_KEY = "student"

@METRIC_REGISTRY.register("math_question_validation_metric")
def validate_math_question(example, prediction, trace=None):
    # TODO: Implement proper validation logic
    return 1 if prediction.question else 0

@METRIC_REGISTRY.register("math_answer_validation_metric")
def validate_math_answer(example, prediction, trace=None):
    # TODO: Implement proper validation logic
    return 1 if str(prediction) == str(example.answer) else 0

class GenerateQuestion(dspy.Signature):
    """Given a target number, create a challenging math question with the target number as the answer. Make sure not to include the answer in the question."""
    target: int = dspy.InputField(desc="target number")
    question: str = dspy.OutputField(desc="math question with the target number as the answer")

class AnswerQuestion(dspy.Signature):
    """Given a challenging math question, give the answer to the question as a number only"""
    question: str = dspy.InputField(desc="math question")
    answer: int = dspy.OutputField(desc="answer to the math question (integer)")

class GenerateQuestionModule(dspy.Module):
    def __init__(self, module=dspy.ChainOfThought):
        super().__init__()
        self.cot_question = module(GenerateQuestion)

    def forward(self, target):
        cot_out = self.cot_question(target=target)
        return cot_out

class AnswerQuestionModule(dspy.Module):
    def __init__(self, module=dspy.ChainOfThought):
        super().__init__()
        self.cot_answer = module(AnswerQuestion)

    def forward(self, question):
        cot_out = self.cot_answer(question=question)
        return cot_out


@PLAYER_REGISTRY.register("mathquiz", "mathquiz_teacher")
class MathQuizTeacher(Player):    
    def init_actions(self):
        return {
            "GenerateQuestion": GenerateQuestionModule(module=dspy.ChainOfThought),
            "AnswerQuestion": AnswerQuestionModule(module=dspy.ChainOfThought)
        }

@PLAYER_REGISTRY.register("mathquiz", "mathquiz_teacher_predict")
class MathQuizTeacherPredict(Player):
    def init_actions(self):
        return {
            "GenerateQuestion": GenerateQuestionModule(module=dspy.Predict),
            "AnswerQuestion": AnswerQuestionModule(module=dspy.Predict)
        }

@PLAYER_REGISTRY.register("mathquiz", "mathquiz_student")
class MathQuizStudent(Player):
    def init_actions(self):
        return {"AnswerQuestion": AnswerQuestionModule(module=dspy.ChainOfThought)}

@PLAYER_REGISTRY.register("mathquiz", "mathquiz_student_predict")
class MathQuizStudentPredict(Player):
    def init_actions(self):
        return {"AnswerQuestion": AnswerQuestionModule(module=dspy.Predict)}
