import asyncio
import logging

from app.pipeline import VoiceAssistantPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("orchestrator")


async def main():
    logger.info("Starting Voice Assistant Orchestrator")

    pipeline = VoiceAssistantPipeline()

    try:
        await pipeline.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())
