pip install torch==2.1.0 && \
pip install tensorboard==2.11.0  && \
pip install packaging  && \
MAX_JOBS=4 pip install flash-attn==2.3.6 --no-build-isolation && \
pip install -r requirements.txt
