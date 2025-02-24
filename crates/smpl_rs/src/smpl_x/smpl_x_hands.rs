use enum_map::EnumMap;
use ndarray as nd;

// 0.17.1

use crate::common::pose_hands::{HandPair, HandType};

use super::smpl_x::NUM_HAND_JOINTS;

pub struct SmplXHands {
    pub type2pose: EnumMap<HandType, HandPair>, // maps from the handtype to the pose of the two hands
}
impl Default for SmplXHands {
    #[allow(clippy::too_many_lines)]
    fn default() -> Self {
        #[allow(clippy::unreadable_literal)]
        let left_hand_relaxed = nd::Array2::<f32>::from_shape_vec(
            (NUM_HAND_JOINTS, 3),
            vec![
                -0.0423, 0.0847, -0.2358, //
                0.1241, -0.0303, -0.5961, //
                -0.0841, -0.0879, -0.0305, //
                -0.2895, -0.0021, -0.6499, //
                0.0655, 0.0457, -0.1250, //
                0.0112, -0.0282, -0.0638, //
                -0.3319, -0.2805, -0.4219, //
                -0.3363, -0.0655, -0.2898, //
                -0.0872, -0.0354, -0.3104, //
                -0.1407, -0.1648, -0.7088, //
                -0.1462, -0.0815, -0.1202, //
                -0.0698, 0.0568, -0.1203, //
                0.2577, 0.1510, -0.0714, //
                -0.0241, -0.0153, 0.0633, //
                0.2703, 0.1708, -0.2459, //
            ],
        )
        .unwrap();

        #[allow(clippy::unreadable_literal)]
        let left_hand_curled = nd::Array2::<f32>::from_shape_vec(
            (NUM_HAND_JOINTS, 3),
            vec![
                0.0591, 0.0117, -0.4014, //
                0.1519, -0.0621, -0.8565, //
                -0.1639, -0.1507, -0.1877, //
                -0.3072, 0.0422, -0.5385, //
                -0.0167, 0.0520, -0.7455, //
                0.0346, -0.1012, -0.4068, //
                -0.7777, -0.0944, -0.5858, //
                -0.0486, -0.0812, -0.4150, //
                -0.2891, 0.0228, -0.3434, //
                -0.3063, -0.1109, -0.6556, //
                -0.2111, -0.0969, -0.6347, //
                -0.1052, -0.0272, -0.4207, //
                0.5913, 0.3559, -0.2171, //
                -0.1058, -0.0210, 0.3505, //
                0.3081, 0.0842, -0.4556, //
            ],
        )
        .unwrap();

        #[allow(clippy::unreadable_literal)]
        #[allow(clippy::excessive_precision)]
        let left_hand_fist = nd::Array2::<f32>::from_shape_vec(
            (NUM_HAND_JOINTS, 3),
            vec![
                -0.17432889342308044,
                0.2899666428565979,
                -1.682339072227478, //
                0.7156379222869873,
                -0.06946949660778046,
                -1.6961337327957153, //
                -0.5035868883132935,
                -0.06818924844264984,
                -0.36969465017318726, //
                -0.4608190059661865,
                0.11983983218669891,
                -2.01084303855896, //
                -0.111131951212883,
                0.24615702033042908,
                -0.8789557814598083, //
                -0.12609277665615082,
                0.18146048486232758,
                -1.2574352025985718, //
                -1.0199159383773804,
                -0.8440185189247131,
                -1.488898754119873, //
                -0.8425631523132324,
                -0.17709890007972717,
                -0.49536100029945374, //
                -0.6223214864730835,
                0.12810452282428741,
                -0.7870318293571472, //
                -0.3238529860973358,
                -0.5304823517799377,
                -1.7103344202041626, //
                -0.644635796546936,
                -0.325455904006958,
                -1.0210907459259033, //
                -0.15834343433380127,
                -0.04439852386713028,
                -1.03395676612854, //
                0.9578915238380432,
                0.24371644854545593,
                -0.25406309962272644, //
                0.22801567614078522,
                0.4293811023235321,
                -0.06814373284578323, //
                0.5744239091873169,
                0.638904869556427,
                -0.679925262928009, //
            ],
        )
        .unwrap();

        let right_hand_relaxed = Self::left2right_hand(&left_hand_relaxed);
        let right_hand_curled = Self::left2right_hand(&left_hand_curled);
        let right_hand_fist = Self::left2right_hand(&left_hand_fist);

        //make all pairs
        let flat_pair = HandPair {
            left: nd::Array2::<f32>::zeros((NUM_HAND_JOINTS, 3)),
            right: nd::Array2::<f32>::zeros((NUM_HAND_JOINTS, 3)),
        };
        let relaxed_pair = HandPair {
            left: left_hand_relaxed,
            right: right_hand_relaxed,
        };
        let curled_pair = HandPair {
            left: left_hand_curled,
            right: right_hand_curled,
        };
        let fist_pair = HandPair {
            left: left_hand_fist,
            right: right_hand_fist,
        };

        let mut type2pose: EnumMap<HandType, HandPair> = EnumMap::default();
        type2pose[HandType::Flat] = flat_pair;
        type2pose[HandType::Relaxed] = relaxed_pair;
        type2pose[HandType::Curled] = curled_pair;
        type2pose[HandType::Fist] = fist_pair;

        Self { type2pose }
    }
}

impl SmplXHands {
    pub fn left2right_hand(left_hand_pose: &nd::Array2<f32>) -> nd::Array2<f32> {
        let mut right_hand_pose = left_hand_pose.clone();
        right_hand_pose.column_mut(1).map_inplace(|x| *x = -*x);
        right_hand_pose.column_mut(2).map_inplace(|x| *x = -*x);
        right_hand_pose
    }
}
