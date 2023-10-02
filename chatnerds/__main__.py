from .config import Config
from .cli import app

# load and initialise config (Create nerds directory, etc)
global_config = Config.environment_instance()
global_config.bootstrap()

if __name__ == "__main__":
    app()
