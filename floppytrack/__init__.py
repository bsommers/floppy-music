# Copyright 2024 Bill Sommers
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

"""floppytrack -- retro floppy-drive sound-chip synthesizer.

Synthesizes 3.5"/5.25" floppy drive sound effects (spinning motor, stepper
motor ticks, read/write head blips) and plays songs described in a simple
text format, rendered to .wav (8-bit crushed) and .mp3 in ``output/``.
"""

__all__ = ["synth", "songspec", "render"]
__version__ = "0.1.0"
