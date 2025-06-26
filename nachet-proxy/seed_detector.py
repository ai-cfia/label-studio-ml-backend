from PIL import Image
from collections import namedtuple
from urllib.request import Request, urlopen
import io
import base64
import json
from urllib.error import URLError


class ModelAPIError(Exception):
    pass


class SeedDetectorModelAPIError(ModelAPIError):
    pass


class SeedDetectorModel:
    content_type: str
    api_key: str
    deployment_platform: str
    name: str
    endpoint: str


async def request_inference_from_seed_detector(
    model: SeedDetectorModel, previous_result: str
):
    """
    Requests inference from the seed detector model using the previously provided result.

    Args:
        model (SeedDetectorModel): The seed detector model.
        previous_result (str): The previous result used for inference. url encoded base64 image

    Returns:
        dict: A dictionary containing the result JSON and the images generated from the inference.

    Raises:
        ProcessInferenceResultsError: If an error occurs while processing the request.
    """
    try:
        headers = {
            "Content-Type": model.content_type,
            "Authorization": ("Bearer " + model.api_key),
            model.deployment_platform: model.name,
        }

        data = {
            "input_data": {
                "columns": ["image"],
                "index": [0],
                "data": [previous_result],
            }
        }

        body = str.encode(json.dumps(data))
        req = Request(model.endpoint, body, headers, method="POST")
        # req = Request("http://192.168.x.x:12380/score", body, headers, method="POST")
        response = urlopen(req)

        result = response.read()
        result_object = [json.loads(result.decode("utf8"))]
        print(
            json.dumps(result_object[0].get("boxes"), indent=4)
        )  # TODO Transform into logging

        return {
            "result_json": result_object,
            # "images": process_image_slicing(previous_result, result_object)
        }
    except (
        KeyError,
        TypeError,
        IndexError,
        ValueError,
        URLError,
        json.JSONDecodeError,
    ) as error:
        print(error)
        raise SeedDetectorModelAPIError(
            f"Error while processing inference results :\n {str(error)}"
        ) from error
