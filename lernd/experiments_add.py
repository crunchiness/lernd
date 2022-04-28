from lernd.classes import LanguageModel, ProgramTemplate, ILP
from lernd.lernd_types import Constant, RuleTemplate
from lernd.util import str2pred, str2ground_atom


def setup_even_noisy():
    print('Setting up "Even" problem...')
    # Language model
    target_pred = str2pred('even/1')
    zero_pred = str2pred('zero/1')
    succ_pred = str2pred('succ/2')
    preds_ext = [zero_pred, succ_pred]
    constants = [Constant(str(i)) for i in range(11)]
    language_model = LanguageModel(target_pred, preds_ext, constants)

    # Program template
    aux_pred = str2pred('pred/2')
    aux_preds = [aux_pred]
    rules = {
        target_pred: (RuleTemplate(0, False), RuleTemplate(1, True)),
        aux_pred: (RuleTemplate(1, False), None)
    }
    forward_chaining_steps = 6
    program_template = ProgramTemplate(aux_preds, rules, forward_chaining_steps)

    # ILP problem
    ground_zero = str2ground_atom('zero(0)')
    background = [ground_zero] + [str2ground_atom(f'succ({i},{i + 1})') for i in range(10)]
    positive = [str2ground_atom(f'even({i})') for i in [0, 2, 4, 6, 8]]
    negative = [str2ground_atom(f'even({i})') for i in [1, 3, 5, 7, 9, 10]]

    ilp_problem = ILP('even_noisy', language_model, background, positive, negative)
    return ilp_problem, program_template


def s():
    with tf.device('/CPU:0'):
        if args.problem == 'predecessor':
            ilp_problem, program_template = setup_predecessor()
            steps = 100
            mini_batch = 1.0  # no mini batching
            worlds = False
        elif args.problem == 'even':
            ilp_problem, program_template = setup_even()
            steps = 300
            mini_batch = 0.3  # loss is based on 30% of random given examples
            worlds = True
        print_BPN(ilp_problem)
        main_loop(ilp_problem, program_template, steps=steps, mini_batch=mini_batch, worlds=worlds, plot_loss=True, save_output=True)