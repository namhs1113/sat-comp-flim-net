
import argparse
from solver.adversarial_solver import AdversarialSolver


def main(config):

    solver = AdversarialSolver(config)

    # Train and sample the images
    if config.phase == "train":
        solver.train()
    elif config.phase == "test":
        solver.test()


def set_config(phase="train", device_ids=[0, 1, 2, 3], dataset_path="", model_path="",
               num_input_ch=2, num_output_ch=1, batch_size=40,
               adv_weight=1, l1_weight=1, satcomp_weight=1, lb_weight=0.1, l2_penalty=0.001, init_lt=dict()):

    parser = argparse.ArgumentParser()

    # Select phase
    parser.add_argument("--phase", type=str, default=phase)  # Phase - train or test or no_mask
    parser.add_argument("--num_epochs", type=int, default=500)  # Total number of epochs

    # Device setup
    parser.add_argument("--device_ids", type=list, default=device_ids)

    # Path to load/save the trained model
    parser.add_argument("--dataset_path", type=str, default=dataset_path)  # Path to load the train/valid datasets
    parser.add_argument("--model_path", type=str, default="results/" + model_path + "/")
    parser.add_argument("--model_name", type=str, default="model.pth")

    # Size paramters
    parser.add_argument("--num_input_ch", type=int, default=num_input_ch)  # The number of input channels
    parser.add_argument("--num_output_ch", type=int, default=num_output_ch)  # The number of output channels
    parser.add_argument("--batch_size", type=int, default=batch_size)  # The number of mini-batch
    parser.add_argument("--num_workers", type=int, default=16)  # The number of workers to generate the images

    # Hyper-parameters
    parser.add_argument("--adv_weight", type=float, default=adv_weight)  # Weight value for adversarial loss
    parser.add_argument("--l1_weight", type=float, default=l1_weight)  # Weight value for L1 loss
    parser.add_argument("--satcomp_weight", type=float, default=satcomp_weight)  # Weight value for sat-comp pixels
    parser.add_argument("--lb_weight", type=float, default=lb_weight)  # Weight value for lower-bound constraint
    parser.add_argument("--l2_penalty", type=float, default=l2_penalty)  # L2 penalty for L2 regularization
    parser.add_argument("--lr_opt", type=dict, default={_type: {"policy": "plateau",
                                                                "init": init_lt[_type],  # Initial lr
                                                                "term": 2e-7,  # Terminating lr condition
                                                                "gamma": 0.1,  # lr decay level
                                                                "patience": 25  # Plateau length
                                                                } for _type in init_lt.keys()})
    _config_ = parser.parse_args()

    return _config_


if __name__ == "__main__":

    config_ = set_config(phase="train",
                         device_ids=[0, ],
                         dataset_path="dataset/",
                         model_path="test/test_model",
                         batch_size=16, l2_penalty=1e-4,
                         adv_weight=1, l1_weight=10, lb_weight=0.1, satcomp_weight=1000,
                         init_lt={"gen": 2e-4, "dsc": 2e-4})

    main(config_)
