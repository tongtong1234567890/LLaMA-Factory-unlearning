from llmtuner import Evaluator
from llmtuner.hparams import get_eval_args
from llmtuner.eval.inference import InferenceEngine
from typing import Any, Dict, List, Optional


def main(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, eval_args, finetuning_args = get_eval_args(args)
    evaluator = InferenceEngine(model_args, data_args, eval_args, finetuning_args)
    evaluator.eval_tasks()


if __name__ == "__main__":
    main()
