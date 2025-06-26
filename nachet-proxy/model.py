from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from seed_detector import request_inference_from_seed_detector, SeedDetectorModel
import requests
import base64


class NachetDetectorModel(LabelStudioMLBase):
    """Custom ML Backend model"""

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "0.0.1")

    def download_image(self, image_path: str) -> bytes:
        """Download image from path"""
        return requests.get(image_path).content

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Write your inference logic here
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f"""\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}""")

        for task in tasks:
            image_path = task["data"]["image"]
            # download image
            image_bytes = self.download_image(image_path)
            # encode image to base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            model = SeedDetectorModel(
                content_type="application/json",
                api_key="",
                deployment_platform="azureml",
                name="seed-detector",
                endpoint="https://nachet-seed-detector.eastus.azureml.io/score",
            )
            # request inference
            result = request_inference_from_seed_detector(model, image_base64)
            # add result to task
            task["predictions"] = [
                PredictionsModel(
                    created_ago="3 hours",
                    model_version="model 1",
                    result=[
                        {
                            "from_name": "tag",
                            "id": "t5sp3TyXPo",
                            "source": "$image",
                            "to_name": "img",
                            "type": "rectanglelabels",
                            "value": {
                                "height": x["box"]["bottomY"] - x["box"]["topY"],
                                "width": x["box"]["bottomX"] - x["box"]["topX"],
                                "x": x["box"]["topX"],
                                "y": x["box"]["topY"],
                                "rectanglelabels": ["seed"],
                                "rotation": 0,
                            },
                        }
                        for x in result["result_json"][0]["boxes"]
                    ],
                )
            ]

        return ModelResponse(predictions=[x.get_predictions() for x in tasks])

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        # store new data to the cache
        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f"New data: {self.get('my_data')}")
        print(f"New model version: {self.get('model_version')}")

        print("fit() completed successfully.")


class TaskModel:
    def __init__(self, task: Dict):
        self.id = task["id"]
        self.data = task["data"]
        self.project = task["project"]
        self.annotations = task["annotations"]
        self.result = task["result"]
        self.drafts = task["drafts"]
        self.predictions = task["predictions"]
        self.updated_at = task["updated_at"]

    def get_result(self):
        return self.result

    def get_annotations(self):
        return self.annotations

    def get_predictions(self):
        return self.predictions

    def get_updated_at(self):
        return self.updated_at

    def get_data(self):
        return self.data

    def get_project(self):
        return self.project

    def get_drafts(self):
        return self.drafts

    def get_id(self):
        return self.id

    def get_created_at(self):
        return self.created_at


class ContextModel:
    def __init__(self, context: Dict):
        self.annotation_id = context["annotation_id"]
        self.draft_id = context["draft_id"]
        self.user_id = context["user_id"]
        self.result = context["result"]

    def get_annotation_id(self):
        return self.annotation_id

    def get_draft_id(self):
        return self.draft_id

    def get_user_id(self):
        return self.user_id

    def get_result(self):
        return self.result

class PredictionsModel:
    def __init__(self, prediction: Dict):
        self.created_ago = prediction["created_ago"]
        self.model_version = prediction["model_version"]
        # You can create objects inline in Python using dicts or namedtuples, for example.
        # Here, let's assume you want to store the result as a list of dicts (since the sample data shows a list).
        self.result = [
            {
                "from_name": r["from_name"],
                "id": r["id"],
                "source": r["source"],
                "to_name": r["to_name"],
                "type": r["type"],
                "value": {
                    "height": r["value"]["height"],
                    "rectanglelabels": r["value"]["rectanglelabels"],
                    "rotation": r["value"]["rotation"],
                    "width": r["value"]["width"],
                    "x": r["value"]["x"],
                    "y": r["value"]["y"],
                },
            }
            for r in prediction["result"]
        ]

    def get_created_ago(self):
        return self.created_ago

    def get_model_version(self):
        return self.model_version

    def get_result(self):
        return self.result
