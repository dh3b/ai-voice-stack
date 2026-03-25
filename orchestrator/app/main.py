import asyncio
import logging
from app.pipeline import VoiceAssistantPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Voice Assistant Orchestrator...")
    
    pipeline = VoiceAssistantPipeline()
    
    try:
        await pipeline.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())
