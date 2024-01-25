1. 测试模型压缩是否成功（根据关键神经元进行模型压缩）
    ```
    model = DQN.load(f"{Root_dir}/model/AV_model/highway_dqn/20000steps/model.zip")
    sets = deserializer(f"{Root_dir}/output/sampler_result/highway_dqn1/action5/top-3-node-set.data")
    sets = {0:{0, 1, 2}, 1:{0, 1, 2}}
    new_QNet = Compressed_QNetwork(len(sets[0]), len(sets[1]))
    new_QNet.compress(sets[0], sets[1], model.policy.q_net)
    new_QNet.to('cuda:0')

    print("old weight 0", model.policy.q_net.q_net[0].weight.data[:3, :])
    print("new weight 0", new_QNet.q_net[0].weight.data)

    print("old bias 0", model.policy.q_net.q_net[0].bias.data[:3])
    print("new bias 0", new_QNet.q_net[0].bias.data)

    print("old weight 2", model.policy.q_net.q_net[2].weight.data[:3, :3])
    print("new weight 2", new_QNet.q_net[2].weight.data)

    print("old bias 2", model.policy.q_net.q_net[2].bias.data[:3])
    print("new bias 2", new_QNet.q_net[2].bias.data)

    print("old weight 4", model.policy.q_net.q_net[4].weight.data[:, :3])
    print("new weight 4", new_QNet.q_net[4].weight.data)

    print("old bias 4", model.policy.q_net.q_net[4].bias.data)
    print("new bias 4", new_QNet.q_net[4].bias.data)
    ```
2. 