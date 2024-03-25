from chatnerds.config import Config

# load and initialise config (Create nerds directory, etc)
Config.environment_instance().bootstrap()


def run_cli():
    from chatnerds.cli.cli import app

    app()


if __name__ == "__main__":
    run_cli()
