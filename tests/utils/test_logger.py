import logging

from src.utils.logger import Logger


def test_logging_output(caplog):
    with caplog.at_level("INFO"):
        Logger().info("Test info!")

    assert "Test info!" in caplog.text
    assert any(r.levelname == "INFO" for r in caplog.records)


def test_logging_output_lvl_debug(caplog):
    Logger()._logger.setLevel(logging.DEBUG)

    with caplog.at_level("DEBUG"):
        Logger().debug("Test debug!")
        Logger().info("Test info!")
        Logger().warning("Test warning!")
        Logger().error("Test error!")

    assert "Test debug!" in caplog.text
    assert "Test info!" in caplog.text
    assert "Test warning!" in caplog.text
    assert "Test error!" in caplog.text


def test_logging_output_lvl_info(caplog):
    Logger()._logger.setLevel(logging.INFO)

    with caplog.at_level("INFO"):
        Logger().debug("Test debug!")
        Logger().info("Test info!")
        Logger().warning("Test warning!")
        Logger().error("Test error!")

    assert "Test debug!" not in caplog.text
    assert "Test info!" in caplog.text
    assert "Test warning!" in caplog.text
    assert "Test error!" in caplog.text


def test_logging_output_lvl_warn(caplog):
    Logger()._logger.setLevel(logging.WARNING)

    with caplog.at_level("WARNING"):
        Logger().debug("Test debug!")
        Logger().info("Test info!")
        Logger().warning("Test warning!")
        Logger().error("Test error!")

    assert "Test debug!" not in caplog.text
    assert "Test info!" not in caplog.text
    assert "Test warning!" in caplog.text
    assert "Test error!" in caplog.text


def test_logging_output_lvl_error(caplog):
    logger = Logger()
    logger._logger.setLevel(logging.ERROR)

    with caplog.at_level("ERROR"):
        logger.debug("Test debug!")
        logger.info("Test info!")
        logger.warning("Test warning!")
        logger.error("Test error!")

    assert "Test debug!" not in caplog.text
    assert "Test info!" not in caplog.text
    assert "Test warning!" not in caplog.text
    assert "Test error!" in caplog.text
