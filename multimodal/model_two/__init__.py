from .SDL_N_c1 import SDL
from .SDL_N_ms import SDL_ms
from .SDL_N_msg import SDL_msg
def model_generator(method, opt):
    if method == 'SDL':
        model = SDL(dim=opt.dim, band=opt.band).cuda()
    elif method == 'SDL_ms':
        model = SDL_ms(dim=opt.dim, band=opt.band).cuda()
    elif method == 'SDL_msg':
        model = SDL_msg(dim=opt.dim, band=opt.band).cuda()
    return model
