import numpy as np

class DdbgResults( object ):

    def __init__(
            self,
            centroid_embeddings: np.ndarray,
            mislabel_scores: np.ndarray,
            self_influence_scores: np.ndarray,
            final_predicted_classes: np.ndarray,
            top_proponents: np.ndarray = None,
            top_opponents: np.ndarray = None,
    ):

        self.centroid_embeddings = centroid_embeddings
        self.mislabel_scores = mislabel_scores
        self.self_influence_scores = self_influence_scores
        self.final_predicted_classes = final_predicted_classes
        self.top_proponents = top_proponents
        self.top_opponents = top_opponents

    def save( self, output_path: str ):
        with open( output_path, 'wb' ) as fp:
            np.save( fp, self.centroid_embeddings )
            np.save( fp, self.mislabel_scores )
            np.save( fp, self.self_influence_scores )
            np.save( fp, self.final_predicted_classes )
            np.save( fp, self.top_proponents )
            np.save( fp, self.top_opponents )

    @classmethod
    def load( klass, load_path: str ):
        with open( load_path, 'rb' ) as fp:
            centroid_embeddings = np.load( fp )
            mislabel_scores = np.load( fp )
            self_influence_scores = np.load( fp )
            final_predicted_classes = np.load( fp )
            top_proponents = np.load( fp )
            top_opponents = np.load( fp )

        return klass(
            centroid_embeddings = centroid_embeddings,
            mislabel_scores = mislabel_scores,
            self_influence_scores = self_influence_scores,
            final_predicted_classes = final_predicted_classes,
            top_proponents = top_proponents,
            top_opponents = top_opponents
        )
