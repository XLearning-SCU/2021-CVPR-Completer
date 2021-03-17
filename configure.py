def get_default_config(data_name):
    if data_name in ['Caltech101-20']:
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )

    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[20, 1024, 1024, 1024, 128],
                arch2=[59, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )

    elif data_name in ['NoisyMNIST']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 64],
                arch2=[784, 1024, 1024, 1024, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=0,
                start_dual_prediction=100,
                epoch=500,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )

    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 64],
                arch2=[40, 1024, 1024, 1024, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0,
                seed=3,
                start_dual_prediction=100,
                epoch=500,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    else:
        raise Exception('Undefined data_name')
