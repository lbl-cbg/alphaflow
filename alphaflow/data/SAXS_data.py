import pandas as pd
import numpy as np

FeatureDict = Mapping[str, np.ndarray]

def make_saxs_features(
    saxs: np.ndarray, description: str, num_res : int
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=np.object_
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    return features

def parse_saxs(saxs_path: str):
# Here we assume the wrapper will provide a full path.
# We need to figure out how to do that later
    data = pd.read_csv(saxs_path)
    return np.pad(data['P(r)'].values[1:], (0, 512-len(data['P(r)'].values[1:])),constant_values=(0,0))
def process_saxs(
    self,
    saxs_path: str,
    alignment_dir: str,
    alignment_index: Optional[str] = None,
) -> FeatureDict:
    """Assembles features for a single P(r) curve in a CSV file""" 
    input_saxs = parse_saxs(saxs_path)

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res,
    )
    
    return {
        **sequence_features,
        **msa_features, 
        **template_features
    }