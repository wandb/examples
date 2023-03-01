import wandb
import pymsteams
from tenacity import Retrying, stop_after_attempt, wait_random_exponential

config = {
    "webhook_url": "https://sysadminwandb.webhook.office.com/webhookb2/1fc24ff9-fe1f-4cf2-ab3b-88d0189149fd@af722783-84b6-4adc-9c49-c792786eab4a/IncomingWebhook/11767fa7c40840e4bb52817bce8d59aa/44f0d0e4-adc1-4ca6-84bc-1768a8278b1f",
    "text": "hello world",
    "title": "hello title",
    "link_button": {"text": "this is a button", "url": "https://www.google.com"},
    "color": "#FF0000",
}


with wandb.init(config=config, job_type="teams-webhook") as run:
    for attempt in Retrying(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=60),
    ):
        with attempt:
            print("Attempting to send message to Teams...")
            msg = pymsteams.connectorcard(run.config.webhook_url)
            msg.text(run.config.text)
            msg.addLinkButton(
                run.config.link_button["text"],
                run.config.link_button["url"],
            )
            msg.color(run.config.color)
            msg.send()

    run.log_code()
