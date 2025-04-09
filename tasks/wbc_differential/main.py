from argparse import ArgumentParser
from pathlib import Path

import torch
import mlflow

from train import Mode, Tracker, TrackerFactory, train
from classification import ClassificationMode
from localization import LocalizationMode


def main(mode_name: str,
         dataset_path: str,
         model_name: str,
         model_version_major: int,
         model_version_minor: int,
         batch_size: int,
         learning_rate: float,
         epochs: int,
         device: str,
         tracker_list: str):

    trackers: list[Tracker] = []
    tracker_factory = TrackerFactory(mode_name)
    for tracker_name in tracker_list.split(','):
        tracker = tracker_factory.make(tracker_name)
        trackers.append(tracker)

    for tracker in trackers:
        # Note: all the other parameters are logged in train(), but
        # the batch size isn't visible in the train function so we
        # log it here.
        tracker.log_param('batch_size', batch_size)

    mode: Mode | None = None
    match mode_name:
        case 'classifier':
            mode = ClassificationMode()
        case 'localizer':
            mode = LocalizationMode()
    if mode is None:
        raise RuntimeError(f'Invalid mode "{mode_name}"')

    train_dataset, val_dataset = mode.load_datasets(Path(dataset_path))
    model_filename = f'{model_name}-v{model_version_major}.{model_version_minor}.onnx'
    model = mode.open_model()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    train(model,
          model_filename,
          train_loader,
          val_loader,
          learning_rate,
          epochs,
          device,
          mode,
          trackers)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mode',
                        type=str,
                        help='Which model to train ("localizer" or "classifier")')
    parser.add_argument('--dataset',
                        type=str,
                        default='./data',
                        help='Which dataset to load.')
    parser.add_argument('--model-name',
                        type=str,
                        help='What to name the model.')
    parser.add_argument('--model-version-major',
                        type=int,
                        help='The major version number to assign the model.')
    parser.add_argument('--model-version-minor',
                        type=int,
                        help='The minor version number to assign the model.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='The batch size to use when training.')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=1.0e-4,
                        help='The learning rate to assign the optimizer.')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='The number of epochs to train the model.')
    parser.add_argument('--device',
                        type=str,
                        default='auto',
                        help='The device to train the model on.')
    parser.add_argument('--trackers',
                        type=str,
                        default='console',
                        help='A comma separated list of trackers to use (console,tensorboard,mlflow)')
    parser.add_argument('--mlflow-tracking-uri',
                        type=str,
                        default='http://localhost:5000',
                        help='The URI of the MLflow tracking server')
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    main(mode_name=args.mode,
         dataset_path=args.dataset,
         model_name=args.model_name,
         model_version_major=args.model_version_major,
         model_version_minor=args.model_version_minor,
         batch_size=args.batch_size,
         learning_rate=args.learning_rate,
         epochs=args.epochs,
         device=args.device,
         tracker_list=args.trackers)
