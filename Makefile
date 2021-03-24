.PHONY: svhn.build svhn.run

svhn.build:
	docker build --tag svhn "$(PWD)/svhn/"

svhn.run:
	docker run --interactive --rm --tty --volume="$(PWD)/svhn/:/usr/src/app/" svhn
