#!/usr/bin/env python

from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import urllib
import json
import os
import TESTLSTMMODEL
import Emotional_GloVe
import reverse_geocoder as rg
from flask import Flask
from flask import request
from flask import make_response
import getTweets
import re

# Flask app should start in global layout
app = Flask(__name__)


@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)

    print("Request:")
    print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    # print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

def del_Punctutation(s):
    return re.sub(r'^https?://.[\r\n]', '',s, flags=re.MULTILINE)

def processRequest(req):
    print ("started processing")
    if req.get("queryResult").get("action") != "Askingaboutreport":
        return {}
    res = makeWebhookResult()
    return res


def makeWebhookResult():
    tweets = getTweets.get_tweets('@Ah_Samir1907')
    tweets = [del_Punctutation(tweet[0]) for tweet in tweets]
    speech = Emotional_GloVe.main(tweets)
    print("Response:")
    print(speech)

    return {
        "fulfillmentText" : "testing",


    }
# "fulfillmentMessages": speech, --> list bas lsa format al json

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))

    print ("Starting app on port %d" % port)

    app.run(debug=False, port=port, host='0.0.0.0')
