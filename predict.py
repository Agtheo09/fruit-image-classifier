import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import argparse

from configs.config import MODEL_PATH, CLASSES


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

def load_model():
    model = models.resnet18(
        weights=None
    )
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(
            512,
            len(CLASSES)
        )
    )
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )
    model.to(device)
    model.eval()

    return model


transform = transforms.Compose([
    transforms.Resize(
        (224,224)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

def predict(image_path):
    model = load_model()

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


    sorted_probs, sorted_indices = torch.sort(
        probabilities[0],
        descending=True
    )


    print("-------------------")
    print("Predictions:")
    print()


    for prob, idx in zip(
        sorted_probs,
        sorted_indices
    ):

        print(
            f"{CLASSES[idx.item()]}: {prob.item()*100:.2f}%"
        )


    print("-------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fruit image classification using ResNet18"
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to input image"
    )

    args = parser.parse_args()

    predict(
        args.image
    )