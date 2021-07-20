
# MultiLingual Chatbot

A multilingual chatbot which can communicate in multiple languages (English, Hindi, French, Espanol). We can give source and destination language in input and it will reply accordingly.

## Installation

First, create and activate a virtual environment.

|For Windows|
```sh
virtualenv myenv
myenv\Scripts\activate
```
|For Linux|
```sh
virtualenv myenv
myenv/bin/activate
```

Install the dependencies and run chatbot_verbal file.

```sh
pip install -r requirements.txt
```

## Running Locally

```sh
python chatbot_verbal.py
```

It will ask for source and destination language and we have to select it from the list ('en', 'fr', 'hi', 'es')

After that we can initiate communication with chatbot in terminal.
>Note: It is still in beta and only supports some predefined inputs. 

## Libraries

Following are the Libraries used in Multilingual Chatbot:

| Libraries |

 [gTTS] [gtts]
 [NLTK] [nltk]
 [PyAudio] [pyaudio]
 [Numpy] [numpy]

#### License

MIT


   [gtts]: <https://gtts.readthedocs.io/en/latest/>
   [nltk]: <https://www.nltk.org/>
   [pyaudio]: <https://pypi.org/project/PyAudio/>
   [numpy]: <https://numpy.org/doc/stable/user/index.html>
   