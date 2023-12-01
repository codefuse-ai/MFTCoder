
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

# cargo dep libs build
RUN mkdir rust && mkdir rust/src && mkdir rust/src/bin && mkdir rust/tmp && mkdir rust/tmp/logs
COPY tmp/Cargo.toml /app/rust/
COPY tmp/test.rs /app/rust/src/bin/

RUN cd rust && cargo check --bin test

RUN cd /app
WORKDIR /app
