import argparse


def main(grace_period, split_confidence, tie_threshold):
    print(f"Grace Period: {grace_period}")
    print(f"Split Confidence: {split_confidence}")
    print(f"Tie Threshold: {tie_threshold}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='hyperparameters of the base Hoeffding Trees, this is optional and will be otherwise hard coded')

    parser.add_argument('--gp', type=float, choices=range(10, 201, 10),
                        help='The number of instances a leaf should observe between split attempts. Range of values (10; 200). Step = 10.', default=10)

    parser.add_argument('--sc', type=float, choices=[i * 0.05 for i in range(21)],
                        help='The allowable error in split decision. Range of values (0.0; 1.0). Step = 0.05.', default=0.0)

    parser.add_argument('--t', type=float, choices=[i * 0.05 for i in range(21)],
                        help='Threshold below which a split will be forced to break ties. Range of values (0.0; 1.0). Step = 0.05.', default=0.0)

    args = parser.parse_args()

    main(args.gp, args.sc, args.t)


