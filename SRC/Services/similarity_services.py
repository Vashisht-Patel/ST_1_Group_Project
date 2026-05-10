import random


class SimilarityService:

    def __init__(self, dataframe):

        self.dataframe = dataframe

    # Return sample images from predicted class
    def find_similar_images(
        self,
        predicted_label,
        sample_count=3
    ):

        matching_images = self.dataframe[
            self.dataframe["label"] == predicted_label
        ]

        if matching_images.empty:
            return []

        sample_count = min(sample_count, len(matching_images))

        sampled = matching_images.sample(
            n=sample_count,
            random_state=random.randint(0, 10000)
        )

        return sampled["image_path"].tolist()
