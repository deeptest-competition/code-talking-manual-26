import logging
import sys
import tqdm

def setup_logging(log_to=None, log_stdout=True, name="experiment"):
    """
    Set up a named logger to stdout and optionally to a file.
    
    Parameters
    ----------
    log_to : str | None
        Path to a log file. If None, no file logging is added.
    log_stdout : bool
        Whether to log to stdout.
    name : str
        Name of the logger (can be used to retrieve it in other scripts).
    
    Returns
    -------
    logging.Logger
        The configured named logger.
    """
    # Disable messages from matplotlib
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Create or get a named logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers (if any)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if log_stdout:  # Stream handler (stdout)
        term_handler = logging.StreamHandler(sys.stdout)
        term_handler.setFormatter(formatter)
        logger.addHandler(term_handler)

    # Optional file handler
    if log_to is not None:
        file_handler = logging.FileHandler(log_to, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("Logging setup." + (f" Writing to file: {log_to}" if log_to else ""))

    # Suppress noisy Azure SDK logs
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Disable all tqdm progress bars
    tqdm.tqdm.disable = True
    # Stop propagation to root logger
    logger.propagate = False
    return logger
