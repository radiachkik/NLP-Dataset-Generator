from Generator import Generator


def main():
    generator = Generator(template='templates/master.chatette', output_file='result/dataset.json')
    generator.run()


if __name__ == '__main__':
    main()
