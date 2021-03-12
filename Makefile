.PHONY: svhn

svhn:
	docker build --tag svhn "$(PWD)/svhn/"
	docker run --interactive --rm --tty --volume="$(PWD)/svhn/:/usr/src/app/" svhn
