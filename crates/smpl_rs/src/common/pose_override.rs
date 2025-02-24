use std::collections::HashSet;

use super::{pose_hands::HandType, pose_parts::PosePart};

use ndarray as nd;
use strum::IntoEnumIterator;

/// Component for pose override
#[derive(Clone)]
pub struct PoseOverride {
    pub denied_parts: HashSet<PosePart>,
    pub overwrite_hands: Option<HandType>,
    // if we overwrite the hands, we store here the original
    //TODO maybe store this in the Pose itself, currently it doesn't make much sense to store it here
    pub original_left_hand: Option<nd::Array2<f32>>,
    pub original_right_hand: Option<nd::Array2<f32>>,
}
impl PoseOverride {
    pub fn allow_all() -> Self {
        Self {
            denied_parts: HashSet::new(),
            overwrite_hands: None,
            original_left_hand: None,
            original_right_hand: None,
        }
    }
    pub fn deny_all() -> Self {
        let mut denied_parts = HashSet::new();
        for part in PosePart::iter() {
            denied_parts.insert(part);
        }

        Self {
            denied_parts,
            overwrite_hands: None,
            original_left_hand: None,
            original_right_hand: None,
        }
    }

    #[must_use]
    pub fn allow(mut self, part: PosePart) -> Self {
        self.denied_parts.remove(&part);
        self
    }

    /// Will set the rotation of these joints to zero
    #[must_use]
    pub fn deny(mut self, part: PosePart) -> Self {
        self.denied_parts.insert(part);
        self
    }

    /// Will set the poses of the hands to something else
    #[must_use]
    pub fn overwrite_hands(mut self, hand_type: HandType) -> Self {
        self.set_overwrite_hands(hand_type);
        self
    }

    #[must_use]
    pub fn build(self) -> Self {
        self
    }

    /// Will set the poses of the hands to something else
    pub fn set_overwrite_hands(&mut self, hand_type: HandType) {
        self.overwrite_hands = Some(hand_type);
    }

    pub fn remove_overwrite_hands(&mut self) {
        self.overwrite_hands = None;
    }

    pub fn get_overwrite_hands_type(&self) -> Option<HandType> {
        self.overwrite_hands
    }
}
