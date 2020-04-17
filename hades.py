#!/usr/bin/python

import os
import socket
import logging
import hades_utils
import json
import configparser
import time
from select import select
from os import path

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("failed to import paho.mqtt.client")

def start():
    logging.basicConfig(level=logging.DEBUG)

    # parse the hades config
    config = HadesConfig("hades.conf")
    config.parseConfig()

    # start the client program with parsed config
    client = Hades(config)
    client.main()

    return 0

class HadesConfig:

    def __init__(self, config):
        # Initialize HadesConfig
        self.config = config
        self.parser = configparser.ConfigParser()
        self.mqtt = {}

        # Logging
        # self.log_level = logging.DEBUG

        # MQTT config
        self.mqtt["user"] = "mock"
        self.mqtt["password"] = "test"
        self.mqtt["server"] = "172.18.0.3"
        self.mqtt["port"] = 1883

    def parseConfig(self):
        if self.parser is not None:
            self.parser.read(self.config)
        else:
            logging.error("failed to parse config")
            return

        if self.parser["MQTT"]:
            self.mqtt["user"] = self.parser.get("MQTT", "User")
            self.mqtt["password"] = self.parser.get("MQTT", "Password")
            self.mqtt["server"] = self.parser.get("MQTT", "Server")
            self.mqtt["port"] = self.parser.getint("MQTT", "Port")

    def getMqttConfig(self):
        return self.mqtt


# Hades is a main class for MQTT message handling as well as calling
# the model generator.
class Hades:

    def __init__(self, config):
        self.config = config
        self.client_id="hades"
        self.models_dir="./models"
        self.hermesPrefix = "hermes"

    """on_connect will be called when the MQTT client connects to the MQTT
    broker.
    """
    def on_connect(self, client, userdata, flags, rc):
        print("Hades connected to " + self.config.mqtt["server"])

        # Handle subscriptions
        self.client.subscribe("hades/+/+/+", 0)
        self.client.subscribe("hades/+/+/model/+", 0)

    """subscribe will subscribe all required topics with their respective
    handlers for MQTT.
    """
    def subscribe(self):
        topics = {
            "hades/+/+/statistics": self.on_stats,
            "hades/+/+/ping": self.on_ping,
            "hades/+/+/model/request": self.on_request
        }

        # subscribe to the given topic list
        for topic in topics:
            self.client.message_callback_add(topic, topics[topic])
            logging.info("subscribed to " + topic)

    """on_disconnect will be called when the MQTT client becomes disconnected
    from the broker.
    """
    def on_disconnect(self, client, userdata, rc):
        self.disconnected = True, rc

    def on_message(self, client, userdata, msg):
        print(msg.topic)

    """on_stats will handle the received messages of a devices statistics,
    when data is received, it will send this data for analyze.

    endpoint: /hades/+/+/statistics
    """
    def on_stats(self, client, userdata, msg):
        # json.loads(msg.payload)
        _, net, mac, _ = hades_utils.split_segments4(msg.topic)

        if not hades_utils.verify_mac(mac):
            logging.info("MAC address (%s) is invalid!", mac)

        # call keras for model relearning

        return

    """on_request will handle a request for a new model. A server may ask for
    a new model via this handler and the handler should respond with a new
    model.

    endpoint: /hades/+/+/model/request
    """
    def on_request(self, client, userdata, msg):
        _, net, mac, _ = hades_utils.split_segments4(msg.topic)

        if not hades_utils.verify_mac(mac):
            logging.info("MAC address (%s) is invalid!", mac)

        # does a model for this device exist?
        if hades_utils.check_model(self.models_dir, mac):
            modelMac = self.models_dir + "/" + mac + ".tflite"

            # prepare data
            f = open(modelMac, 'rb')
            byteArray = bytes(f.read())
            f.close()

            timeSent = time.localtime()
            currentTime = json.dumps({
                "model": modelMac,
                "time_sent": time.strftime("%H:%M:%S", timeSent)
            })

            # construct publish topics for hermes and iotctl.
            topic = f"{self.hermesPrefix}/node/{net}/{mac}/hades/model/receive"
            topicEvent = f"node/{net}/{mac}/hades/event/sent"

            logging.debug("publishing on %s", topic)
            self.client.publish(topic, byteArray, 0)

            # Notify IoT Controller about a sent model
            # TODO: implement an Event infrastructure.
            logging.debug("publishing on %s", topicEvent)
            self.client.publish(topicEvent, currentTime, 0)
        else:
            logging.info("no model for node (%s)", mac)
        return
    
    """on_ping will handle a request for a ping checking. A device may ask for
    a ping check and we should respond to it.
    """
    def on_ping(self, client, userdata, msg):
        _, net, mac, _ = hades_utils.split_segments4(msg.topic)

        if not hades_utils.verify_mac(mac):
            logging.info("MAC address (%s) is invalid!", mac)
            return

        topic = f"{self.hermesPrefix}/node/{net}/{mac}/hades/pong"
        logging.debug("publishing on %s", topic)
        self.client.publish(topic, None, 0)

    def on_log(self, client, level, buf):
        logging.debug(buf)

    """main shall be the entry point for this function and will setup required
    connections for MQTT broker and other required services.
    """
    def main(self):
        mqttConfig = self.config.mqtt

        # TODO: make distinct init functions for different services.
        self.disconnected = (False, None)
        self.t = time.time()
        self.state = 0

        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_log = self.on_log
        #self.client.enable_logger(logger=logging)

        self.subscribe()

        # Attach handlers
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message

        logging.info("mqtt client connecting to %s:%d", mqttConfig["server"], mqttConfig["port"])

        # Handle connections
        self.client.username_pw_set(mqttConfig["user"], mqttConfig["password"])
        try:
            self.client.connect(mqttConfig["server"], int(mqttConfig["port"]))
        except:
            logging.error("failed to initialize MQTT client")
            self.client = None
            return

        while True:
            self.client.loop_forever()
        return

# Main entry point
if __name__ == '__main__':
    start()
