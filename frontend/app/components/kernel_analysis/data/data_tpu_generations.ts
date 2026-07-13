/** TPU generation class with name info. */
export class TpuGeneration {
  id: string;
  name: string;
  constructor(id: string, name: string) {
    this.id = id;
    this.name = name;
  }
}

/** TPU generations supported for kernel profiling. */
export const TPU_GENERATIONS = [new TpuGeneration('v7x', 'TPU v7x')];
