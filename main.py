
def get_score_from_api(job=None, model=None, model_name=None, verbose=True):
    import json
    import api
    import solutionHelper

    with open('credentials.json', 'r') as f:
        credentials = json.load(f)

    api_key = credentials['api-token']
    image_folder_path = 'api_images'

    if model is None:
        model = load_model(job)
        model_name = job['model_name']

    result = api.init_game(api_key)
    game_id = result["gameId"]
    rounds_left = result['numberOfRounds']

    if verbose:
        print("Starting a new game with id: " + game_id)
        print("The game has {} rounds and {} images per round".format(rounds_left, result["imagesPerRound"]))

    while rounds_left > 0:
        if verbose:
            print("Starting new round, {} rounds left".format(rounds_left))

        solutions = []
        zip_bytes = api.get_images(api_key)
        image_names = solutionHelper.save_images_to_disk(zip_bytes, image_folder_path)
        for name in image_names:
            path = image_folder_path + "/" + name
            image_solution = analyze_image(path, model, model_name=model_name)
            solutions.append({"ImageName": name,
                              "BuildingPercentage": image_solution["building_percentage"],
                              "RoadPercentage": image_solution["road_percentage"],
                              "WaterPercentage": image_solution["water_percentage"]})
        solution_response = api.score_solution(api_key, {"Solutions": solutions})

        if verbose:
            solutionHelper.print_errors(solution_response)
            solutionHelper.print_scores(solution_response)

        total_score = solution_response['totalScore']
        rounds_left = solution_response['roundsLeft']

    # solutionHelper.clean_images_from_folder(image_folder_path)
    return total_score


def analyze_image(image_path, model, model_name):
    from PIL import Image
    from torchvision import transforms
    from config import device
    import torch
    pil_image_2_tensor = transforms.ToTensor()

    pil_image = Image.open(image_path)
    image_tensor = pil_image_2_tensor(pil_image)
    image_tensor = image_tensor.to(device).unsqueeze(0)

    with torch.no_grad():
        if model_name.startswith('perc'):   # doesnt use percentage weight
            out_percentages = model(image_tensor)
            out_bitmap = None
        else:
            out_bitmap, out_percentages = model(image_tensor)

    return_dict = {"building_percentage": out_percentages[0, 2].item() * 100,
                   "water_percentage": out_percentages[0, 1].item() * 100,
                   "road_percentage": out_percentages[0, 3].item() * 100}
    del image_tensor, out_percentages, out_bitmap, pil_image
    return return_dict


def load_model(job):
    from config import device
    # import torch

    model = job['model'].float().to(device).eval()
    model.load_state_dict(job['result']['model_state'])

    return model


def load_model_pth(model_pth):
    from UnetModel import UNet
    from config import device
    import torch

    net = UNet()
    net.load_state_dict(torch.load(model_pth, map_location=device))

    return net

