import logging
import logging.handlers
import os

def setup_logging(logfile, console_handler=True):
    logfile_name = logfile + '.log'
    logfile_object = logfile + '_log'
    rm_logfile = 'rm ' + logfile_name
    os.system(rm_logfile)
    log = logging.getLogger(logfile_object)
    log.setLevel(logging.INFO)
    log.propagate = False
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

    handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", logfile_name))
    handler.setFormatter(formatter)
    log.addHandler(handler)

    if(console_handler == True):
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        log.addHandler(consoleHandler)
    return(log)