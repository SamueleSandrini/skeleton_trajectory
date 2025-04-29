# Copyright 2024 National Research Council STIIMA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from rclpy.publisher import Publisher
from std_msgs.msg import Header


class CustomPublisherBase(ABC):

    @abstractmethod
    def get_msg_type(self):
        pass

    @abstractmethod
    def get_topic(self) -> str:
        pass

    # @abstractmethod
    # def get_qos_profile(self) -> Union[QoSProfile, int]:
    #     pass

    @abstractmethod
    def publish(
        self,
        publisher: Publisher,
        keypoints: Dict[int, np.array],
        header: Header,
    ):
        pass

    @abstractmethod
    def get_message(self, keypoints: Dict[int, np.array], header: Header):
        pass
