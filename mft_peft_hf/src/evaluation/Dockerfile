
# Take bigcode-multiple as basse
FROM evaluation-harness-multiple:latest

# install rustc and cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY . /app

# install boost dev lib
RUN apt-get update && apt-get install -y libboost-all-dev

# install openssl
RUN apt-get update && apt-get install -y libssl-dev

# GOLang
RUN go mod init humanevalpack && go get github.com/stretchr/testify