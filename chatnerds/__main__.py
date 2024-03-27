from chatnerds.config import Config
from chatnerds.cli.cli import app

# load and initialise config (Create nerds directory, etc)
Config.environment_instance().bootstrap()


def run_cli():
    try:
        app()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


if __name__ == "__main__":
    run_cli()
