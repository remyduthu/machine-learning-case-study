.PHONY: svhn.build svhn.run weather.build weather.run

svhn.build:
	docker build --tag svhn "$(PWD)/svhn/"

svhn.run:
	docker run --interactive --rm --tty --volume="$(PWD)/svhn/:/usr/src/app/" svhn

weather.build:
	docker build --tag weather "$(PWD)/weather/"

weather.run:
	docker run --interactive --rm --tty --volume="$(PWD)/weather/:/usr/src/app/" weather
