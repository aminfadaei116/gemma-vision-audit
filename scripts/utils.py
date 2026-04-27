import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()
DTYPE  = torch.bfloat16

MODEL_ID     = "google/gemma-4-E2B-it"
ADAPTER_PATH = "./outputs/gemma4_e2b_artifact_assessor_lora"

USER_PROMPT = (
    "Describe any visual artifacts or physical defects in this AI-generated image. "
    "Be specific about the type, location, and severity of each issue."
)

SYSTEM_PROMPT = (
    "You are an expert computer vision quality analyst specializing in generative model artifacts. "
    "You receive an AI-generated image and structured artifact annotations. "
    "Write a single, detailed natural language description of the visual defects present. "
    "Be specific: mention location, affected body part or object, severity, and artifact type. "
    "If no artifacts are present, confirm that the image looks physically correct."
)

ARTIFACT_TEMPLATES: dict[str, str] = {
    # Anatomy
    "extra_finger":        "The subject's hand has an incorrect number of fingers.",
    "extra_limb":          "The subject has an extra limb that is anatomically incorrect.",
    "missing_limb":        "The subject appears to be missing a limb.",
    "joint_deformity":     "A joint appears unnaturally bent or deformed.",
    "body_proportion":     "The body proportions of the subject are physically implausible.",
    # Attribute
    "color_inconsistency": "The image contains color inconsistencies that are not physically plausible.",
    "texture_artifact":    "The surface texture contains visible generation artifacts.",
    "material_error":      "A material in the image does not match expected physical properties.",
    # Interaction
    "object_overlap":      "Two or more objects appear to merge unnaturally.",
    "spatial_violation":   "An object is placed in a position that violates physical spatial logic.",
}

LABEL_KEYS: list[str] = list(ARTIFACT_TEMPLATES.keys())
