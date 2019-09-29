import api
import solutionHelper
import json
import torch
from UnetModel import UNet
from PIL import Image
from torchvision import transforms

with open('credentials.json', 'r') as f:
    credentials = json.load(f)

api_key = credentials['api-token']
image_folder_path = 'api_images'


pil_image_2_tensor = transforms.ToTensor()



def main():
    model, device = load_net()

    result = api.init_game(api_key)
    game_id = result["gameId"]
    rounds_left = result['numberOfRounds']
    print("Starting a new game with id: " + game_id)
    print("The game has {} rounds and {} images per round".format(rounds_left, result["imagesPerRound"]))
    while rounds_left > 0:
        print("Starting new round, {} rounds left".format(rounds_left))
        solutions = []
        zip_bytes = api.get_images(api_key)
        image_names = solutionHelper.save_images_to_disk(zip_bytes, image_folder_path)
        for name in image_names:
            path = image_folder_path + "/" + name
            image_solution = analyze_image(path, model, device)
            solutions.append({"ImageName": name,
                              "BuildingPercentage": image_solution["building_percentage"],
                              "RoadPercentage": image_solution["road_percentage"],
                              "WaterPercentage": image_solution["water_percentage"]})
        solution_response = api.score_solution(api_key, {"Solutions": solutions})
        solutionHelper.print_errors(solution_response)
        solutionHelper.print_scores(solution_response)
        rounds_left = solution_response['roundsLeft']

    solutionHelper.clean_images_from_folder(image_folder_path)


def analyze_image(image_path, model, device):
    """
    ----------------------------------------------------
    TODO Implement your image recognition algorithm here
    ----------------------------------------------------
    """
    pil_image = Image.open(image_path)
    image_tensor = pil_image_2_tensor(pil_image)
    image_tensor = image_tensor.to(device).unsqueeze(0)
    _, out_percentages = model(image_tensor)

    return_dict = {"building_percentage": out_percentages[0, 2] * 100,
                   "water_percentage": out_percentages[0, 1] * 100,
                   "road_percentage": out_percentages[0, 3] * 100}

    return return_dict


def load_net():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(3, 4).float()
    model = model.to(device)

    model_name = 'test_net.pth'
    model_path = f'models/trained/{model_name}'
    model.load_state_dict(torch.load(model_path))

    return model, device

main()
