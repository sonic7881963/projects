
import { _decorator, Component, Node, resources, Sprite, SpriteFrame, EventTouch, Label, macro, Prefab, instantiate, Vec3, BoxCollider, PhysicsSystem2D, EPhysics2DDrawFlags, director, PhysicsSystem, sys} from 'cc';
import Gloabl from './gloabl';
import Soldier from './soldier';
const { ccclass, property } = _decorator;

/**
 * Predefined variables
 * Name = mainScene
 * DateTime = Mon Feb 14 2022 11:08:04 GMT+0800 (中国标准时间)
 * Author = tc123456
 * FileBasename = mainScene.ts
 * FileBasenameNoExtension = mainScene
 * URL = db://assets/typescript/mainScene.ts
 * ManualUrl = https://docs.cocos.com/creator/3.4/manual/zh/
 *
 */
 
@ccclass('mainScene')
export class mainScene extends Component {
    private static instance: mainScene;
    static js: Boolean = false;
    public static get Instance(): mainScene{
        if(!this.instance){
            this.instance = new mainScene();
        }
        return this.instance;
    }
    exp: number = 0;

    private btnSoldier: Node = null!; 

    private btnHZ: Node = null!; 

    private btnLB: Node = null!; 

    public coinLabel = null!; 

    private expLabel: Label = null!; 

    @property({type: Prefab})
    private soldier: Prefab = null!;

    @property({type: Prefab})
    private heroHz: Prefab = null!;

    @property({type: Prefab})
    private heroGy: Prefab = null!;

    private jz: Node = null!; 

    @property({type: Node})
    public winNode: Node = null!;

    @property({type: Node})
    public btn_close: Node = null!;

    @property({type: Node})
    public kj: Node = null!;

    public arr: Node[] = [];




    onLoad(){

        mainScene.instance = this;
        PhysicsSystem2D.instance.enable = true;
        PhysicsSystem2D.instance.debugDrawFlags = EPhysics2DDrawFlags.Aabb |
        EPhysics2DDrawFlags.Pair |
        EPhysics2DDrawFlags.CenterOfMass |
        EPhysics2DDrawFlags.Joint |
        EPhysics2DDrawFlags.Shape;
        this.coinLabel = this.node.getChildByName("dikuan").getChildByName("Sprite2").getChildByName("num1").getComponent(Label);
        this.btnSoldier = this.node.getChildByName("dikuan").getChildByName("Layout2").getChildByName("sz-0");
        this.btnHZ = this.node.getChildByName("dikuan").getChildByName("Layout2").getChildByName("gj-1");
        this.btnLB = this.node.getChildByName("dikuan").getChildByName("Layout2").getChildByName("zd-1");
        this.expLabel = this.node.getChildByName("dikuan").getChildByName("Sprite2").getChildByName("num2").getComponent(Label);
        this.jz = this.node.getChildByName("jz");
        this.kj = this.node.getChildByName("kj");
    }

    start () {

        //this.solvingUI.active=false;
        //const audioSource = this.node.getComponent(AudioSource)!;
        //this._audioSource = audioSource;
        // let data = sys.localStorage.getItem('userda');
        // if(data){
        //     this.onBtnClose();
        //     return;
        // }
        // sys.localStorage.setItem('userda', JSON.stringify("userda"));
        
        this.chooseHero();
        this.winNode.getChildByName("btn").on(Node.EventType.MOUSE_DOWN, (touch: EventTouch) => {
            console.log('Mouse down');
            this.onBtnClose();
        }, this);
          this.kj.on(Node.EventType.MOUSE_DOWN, (touch: EventTouch) => {
            console.log('Mouse down');
            mainScene.js = true;
            
        
 

        }, this);
        this.btn_close.on(Node.EventType.MOUSE_DOWN, (touch: EventTouch) => {
            console.log('Mouse down');
            this.onBtnClose();
        }, this);
        this.btnSoldier.on(Node.EventType.MOUSE_DOWN, (touch: EventTouch) => {
            console.log('Mouse down');
            if(this.btnSoldier.getComponent(Sprite).spriteFrame.name != 'xb-1'){
                console.log("小兵")
                if(Gloabl.coin >= 5){
                    Gloabl.coin -= 5;
                }
                this.chooseHero();
                this.coinLabel.string = ": " + Gloabl.coin.toString();
                this.createHero(0);
            }
        }, this);
        this.btnHZ.on(Node.EventType.MOUSE_DOWN, (touch: EventTouch) => {
            console.log('Mouse down');
            let sp = this.btnHZ.getComponent(Sprite).spriteFrame
            if(this.btnHZ.getComponent(Sprite).spriteFrame.name != 'hz-1'){
                console.log("黄忠")
                if(Gloabl.coin >= 10){
                    Gloabl.coin -= 10;
                }
                this.chooseHero();
                this.coinLabel.string = ": " + Gloabl.coin.toString();
                this.createHero(1);
            }
        }, this);
        this.btnLB.on(Node.EventType.MOUSE_DOWN, (touch: EventTouch) => {
            console.log('Mouse down');
            if(this.btnLB.getComponent(Sprite).spriteFrame.name != 'gy-1'){
                console.log("吕布")
                if(Gloabl.coin >= 30){
                    Gloabl.coin -= 30;
                }
                this.chooseHero();
                this.coinLabel.string = ": " + Gloabl.coin.toString();
                this.createHero(2);
            }
        }, this);
        this.coinLabel.string = ": " + Gloabl.coin.toString();
        this.expLabel.string = ": " + this.exp.toString();
        this.createSoldier();
    }

    private onBtnClose() {
        this.node.removeFromParent();
        director.getScene().removeFromParent();
        //this._audioSource.stop();
        const queryString = window.location.search;
        console.log("当前地址" + queryString);
        const urlParams = new URLSearchParams(queryString);
        const code = this.getQueryVariable('url');
        console.log("截取的url" + code);
        window.location.href=code;
    }

    private getQueryVariable(variable: string){
        let search = "http://pjax.weisuiyu.com/?url=http://www.baidu.com";
        var query =  window.location.search.substring(1);
        var vars = query.split("&");
        for (var i=0;i<vars.length;i++) {
            var pair = vars[i].split("=");
            return pair[1];
        }
    }

    private chooseHero(): void{
        let url1 = "";
        let url2 = "";
        let url3 = "";
        if(Gloabl.coin >= 5){
            url1 = "xb-0/spriteFrame";
        }else{
            url1 = "xb-1/spriteFrame";
        }
        if(Gloabl.coin >= 10){
            url2 = "hz-0/spriteFrame";
        }else{
            url2 = "hz-1/spriteFrame";
        }
        if(Gloabl.coin >= 30){
            url3 = "gy-0/spriteFrame";
        }else{
            url3 = "gy-1/spriteFrame";
        }
        resources.load(url1, SpriteFrame, (err: any, spriteFrame) => {
            this.btnSoldier.getComponent(Sprite).spriteFrame = spriteFrame;
        });
        resources.load(url2, SpriteFrame, (err: any, spriteFrame) => {
            this.btnHZ.getComponent(Sprite).spriteFrame = spriteFrame;
        });
        resources.load(url3, SpriteFrame, (err: any, spriteFrame) => {
            this.btnLB.getComponent(Sprite).spriteFrame = spriteFrame;
        });
    }

    createSoldier(): void{

        let interval = 2;
        // 开始延时
        let delay = 1;
        let array = [0,0,0,0,1,0,0,1,0,2];
        let index = 0;
        this.schedule(function() {
            // 这里的 this 指向 component
            let prefab = null;
            switch (array[index]) {
                case 0:
                    prefab = this.soldier;
                    break;
                case 1:
                    prefab = this.heroHz;
                    break;
                case 2:
                    prefab = this.heroGy;
                    break;
                default:
                    break;
            }
            if(Gloabl.speed > 0 && index <= 9){
                let item : Node = instantiate(prefab);
                item.name = "enemy";
               
                item.setPosition(new Vec3(1100, 0, 0));
                this.jz.addChild(item);
                if(array[index] == 0){
                    item.setScale(1,1,1);
                }else{
                    item.setScale(-1,1,1);
                }
                index += 1;
            }
            
        }, interval, macro.REPEAT_FOREVER, delay);
    }

    createHero(index: number): void{
        
        // 这里的 this 指向 component
        let prefab = null;
        switch (index) {
            case 0:
                prefab = this.soldier;
                break;
            case 1:
                prefab = this.heroHz;
                break;
            case 2:
                prefab = this.heroGy;
                break;
            default:
                break;
        }
        let item : Node = instantiate(prefab);
        item.name = "hero";
        
        if(index == 0){
            item.setScale(-1,1,1);
        }else{
            item.setScale(1,1,1);
        }
        item.setPosition(new Vec3(200, 0, 0));
        this.jz.addChild(item);
    }

    public refresh(): void{
        this.exp += 10;
        this.coinLabel.string = ": " + Gloabl.coin.toString();
        this.expLabel.string = ": " + this.exp.toString();
        this.chooseHero();
    }
}

/**
 * [1] Class member could be defined like this.
 * [2] Use `property` decorator if your want the member to be serializable.
 * [3] Your initialization goes here.
 * [4] Your update function goes here.
 *
 * Learn more about scripting: https://docs.cocos.com/creator/3.4/manual/zh/scripting/
 * Learn more about CCClass: https://docs.cocos.com/creator/3.4/manual/zh/scripting/ccclass.html
 * Learn more about life-cycle callbacks: https://docs.cocos.com/creator/3.4/manual/zh/scripting/life-cycle-callbacks.html
 */
