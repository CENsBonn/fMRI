import logging
from livereload import Server, shell


def main():
    server = Server()
    files = [
        "*.md",
        "*.yml",
        "images/*",
    ]
    for filename in files:
        server.watch(filename, "make build")

    livereload_logger = logging.getLogger("livereload")

    max_attempts = 10
    port = 8008
    for i in range(max_attempts):
        try:
            server.serve(
                root="_build/html",
                port=port,
                host="localhost",
                open_url_delay=1,
            )
            break
        except OSError:
            if i == max_attempts - 1:
                raise
            port += 1
            livereload_logger.handlers.clear()


if __name__ == "__main__":
    main()
