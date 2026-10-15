class MyTrinConfig:
    def __init__(self,lr=3e-4,bath_size=64,warmup_steps=100):
        self.lr = lr
        self.batch_size = bath_size
        self.warmup_stwps = 100

    def __repr__(self):
        return f"TrainConfig(lr = {self.lr},batch_size = {self.batch_size},...)"
    def __eq__(self, other):
        return (self.lr == other.lr and self.batch_size == other.batch_size)


if __name__ =="__main__":
    print("This is my code")
    mytrain = MyTrinConfig(1e-4,32,1000)

    print(mytrain)



