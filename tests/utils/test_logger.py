import logging

from src.utils.logger import Logger


def test_logging_output(caplog):
    logger = Logger().get_logger()

    with caplog.at_level("INFO"):
        logger.info("Test info!")

    assert "Test info!" in caplog.text
    assert any(r.levelname == "INFO" for r in caplog.records)


def test_logging_output_lvl_debug(caplog):
    logger = Logger().get_logger()

    with caplog.at_level("DEBUG"):
        logger.debug("Test debug!")
        logger.info("Test info!")
        logger.warning("Test warning!")
        logger.error("Test error!")

    assert "Test debug!" in caplog.text
    assert "Test info!" in caplog.text
    assert "Test warning!" in caplog.text
    assert "Test error!" in caplog.text

    assert any(r.levelname == "DEBUG" for r in caplog.records)
    assert any(r.levelname == "INFO" for r in caplog.records)
    assert any(r.levelname == "WARNING" for r in caplog.records)
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_logging_output_lvl_info(caplog):
    logger = Logger(log_lvl=logging.INFO).get_logger()

    with caplog.at_level("INFO"):
        logger.debug("Test debug!")
        logger.info("Test info!")
        logger.warning("Test warning!")
        logger.error("Test error!")

    assert "Test debug!" not in caplog.text
    assert "Test info!" in caplog.text
    assert "Test warning!" in caplog.text
    assert "Test error!" in caplog.text

    assert not any(r.levelname == "DEBUG" for r in caplog.records)
    assert any(r.levelname == "INFO" for r in caplog.records)
    assert any(r.levelname == "WARNING" for r in caplog.records)
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_logging_output_lvl_warn(caplog):
    logger = Logger(log_lvl=logging.WARN).get_logger()

    with caplog.at_level("WARNING"):
        logger.debug("Test debug!")
        logger.info("Test info!")
        logger.warning("Test warning!")
        logger.error("Test error!")

    assert "Test debug!" not in caplog.text
    assert "Test info!" not in caplog.text
    assert "Test warning!" in caplog.text
    assert "Test error!" in caplog.text

    assert not any(r.levelname == "DEBUG" for r in caplog.records)
    assert not any(r.levelname == "INFO" for r in caplog.records)
    assert any(r.levelname == "WARNING" for r in caplog.records)
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_logging_output_lvl_error(caplog):
    logger = Logger(log_lvl=logging.ERROR).get_logger()

    with caplog.at_level("ERROR"):
        logger.debug("Test debug!")
        logger.info("Test info!")
        logger.warning("Test warning!")
        logger.error("Test error!")

    assert "Test debug!" not in caplog.text
    assert "Test info!" not in caplog.text
    assert "Test warning!" not in caplog.text
    assert "Test error!" in caplog.text

    assert not any(r.levelname == "DEBUG" for r in caplog.records)
    assert not any(r.levelname == "INFO" for r in caplog.records)
    assert not any(r.levelname == "WARNING" for r in caplog.records)
    assert any(r.levelname == "ERROR" for r in caplog.records)
