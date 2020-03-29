import os
import socket
import logging
import hades_utils
import json
from time import time
from select import select

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("failed to import paho.mqtt.client")

def start():
    username = ""
    password = ""
    port = 1883
    server = "172.18.0.3"

    logging.basicConfig(level=logging.DEBUG)

    try:
        username = os.environ["IOT_USERNAME"]
        password = os.environ["IOT_PASSWORD"]
    except KeyError:
        username = "hades"
        password = "test"


    client = Hades(server, port)
    client.main()

# Hades is a main class for MQTT message handling as well as calling
# the model generator.
class Hades:
    client_id="hades"

    def __init__(self, server, port):
        self.server = server
        self.port = port
    
    def on_connect(self, client, userdata, flags, rc):
        print("Hades connected to " + self.server)

    """on_disconnect will be called when the MQTT client becomes disconnected
    from the broker.
    """
    def on_disconnect(self, client, userdata, rc):
        self.disconnected = True, rc

    """on_stats will handle the received messages of a devices statistics,
    when data is received, it will send this data for analyze.
    """
    def on_stats(self, client, userdata, msg):
        # json.loads(msg.payload)
        _, net, mac, _ = hades_utils.split_segments4(msg.topic)
        
        if not hades_utils.verify_mac(mac):
            logging.info("MAC address (%s) is invalid!", mac)

    """on_request will handle a request for a new model. A server may ask for
    a new model via this handler.
    """
    def on_request(self, client, userdata, msg):
        print(msg.topic + " received")

    """subscribe will subscribe all required topics with their respective
    handlers for MQTT.
    """
    def subscribe(self):
        topics = {
            "node/+/+/hades/statistics": self.on_stats,
            "node/+/+/hades/model/request": self.on_request
        }

        # subscribe to the given topic list
        for topic in topics:
            self.client.message_callback_add(topic, topics[topic])
            logging.info("subscribed to " + topic)

    def on_log(self, client, level, buf):
        logging.debug(buf)

    """main shall be the entry point for this function and will setup required
    connections for MQTT broker and other required services.
    """
    def main(self):
        # TODO: make distinct init functions for different services.
        self.disconnected = (False, None)
        self.t = time()
        self.state = 0

        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_log = self.on_log
        # self.client.enable_logger(logger=logging)

        # Attach handlers
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

        logging.info("mqtt client connecting to %s:%d", self.server, self.port)

        # Handle connections
        self.client.username_pw_set("mock", "test")
        try:
            self.client.connect(self.server, self.port)

            # Handle subscriptios
            self.subscribe()
            self.client.subscribe("node/+/+/hades/#", 0)
        except:
            logging.error("failed to initialize MQTT client")
            self.client = None

        while True:
            self.client.loop()


start()
