# Copyright 2020-2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# docker build -t safepy
# docker run -it safepy
FROM ubuntu:22.04 as chroot

RUN /usr/sbin/useradd --no-create-home -u 1000 user

RUN apt-get update && apt-get install -y python3 python3-pip git

WORKDIR /home/user

RUN git clone https://github.com/sympy/sympy
WORKDIR /home/user/sympy
# something recent
RUN git checkout 7716f42
RUN python3 setup.py install

COPY flag /
COPY run.sh /home/user/
COPY main.py /home/user/sympy/

CMD /home/user/run.sh